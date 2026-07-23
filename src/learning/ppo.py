from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import multiprocessing as mp
from pathlib import Path
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
    from torch import nn
    from torch.distributions import Categorical, Normal
except ImportError:  # pragma: no cover - exercised only when torch is missing.
    torch = None
    nn = None
    Categorical = None
    Normal = None


@dataclass
class PPOConfig:
    observation_dim: int
    num_actions: int
    hidden_sizes: Tuple[int, ...] = (64, 64)
    learning_rate: float = 5e-5
    gamma: float = 0.99
    gae_lambda: float = 0.97
    clip_ratio: float = 0.2
    train_iters: int = 20
    batch_size: int = 256
    rollout_steps: int = 5000
    target_kl: float = 0.01
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    device: str = "cpu"


def _require_torch() -> None:
    if torch is None:
        raise ImportError("learning.ppo requires PyTorch. Install torch to train PPO policies.")


def _discount_cumsum(values: np.ndarray, discount: float) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float32)
    running = 0.0
    for idx in reversed(range(len(values))):
        running = float(values[idx]) + discount * running
        result[idx] = running
    return result


_MASS_METRIC_BIN_WIDTH_KG = 0.1


def _summarize_mass_metrics(
    episodes: Sequence[Dict[str, float]],
) -> List[Dict[str, float]]:
    bins: Dict[float, List[Dict[str, float]]] = {}
    for episode in episodes:
        mass_kg = float(episode["box_mass_kg"])
        bin_min_kg = (
            math.floor((mass_kg + 1e-9) / _MASS_METRIC_BIN_WIDTH_KG)
            * _MASS_METRIC_BIN_WIDTH_KG
        )
        bins.setdefault(round(bin_min_kg, 6), []).append(episode)

    summaries = []
    for bin_min_kg, bin_episodes in sorted(bins.items()):
        summaries.append(
            {
                "mass_bin_min_kg": bin_min_kg,
                "mass_bin_max_kg": bin_min_kg + _MASS_METRIC_BIN_WIDTH_KG,
                "episodes": float(len(bin_episodes)),
                "success_rate": float(
                    np.mean([episode["success"] for episode in bin_episodes])
                ),
                "slip_rate": float(
                    np.mean([episode["slipped"] for episode in bin_episodes])
                ),
                "mean_desired_force_n": float(
                    np.mean(
                        [
                            episode["mean_desired_force_n"]
                            for episode in bin_episodes
                        ]
                    )
                ),
                "mean_measured_force_n": float(
                    np.mean(
                        [
                            episode["mean_measured_force_n"]
                            for episode in bin_episodes
                        ]
                    )
                ),
                "mean_efficiency_reward": float(
                    np.mean(
                        [
                            episode["efficiency_reward"]
                            for episode in bin_episodes
                        ]
                    )
                ),
            }
        )
    return summaries


def _print_mass_metrics(mass_metrics: Sequence[Dict[str, float]]) -> None:
    for metric in mass_metrics:
        print(
            "mass_bin_kg=[{mass_bin_min_kg:.1f},{mass_bin_max_kg:.1f}) "
            "episodes={episodes:.0f} success_rate={success_rate:.3f} "
            "slip_rate={slip_rate:.3f} "
            "mean_desired_force_n={mean_desired_force_n:.3f} "
            "mean_measured_force_n={mean_measured_force_n:.3f} "
            "mean_efficiency_reward={mean_efficiency_reward:.3f}".format(
                **metric
            )
        )


if torch is not None:

    class SquashedNormal:
        """Independent tanh-squashed Normal components on the interval (-1, 1)."""

        def __init__(self, mean: "torch.Tensor", std: "torch.Tensor") -> None:
            self.base_distribution = Normal(mean, std)

        @property
        def mean(self) -> "torch.Tensor":
            return torch.tanh(self.base_distribution.mean)

        def sample(self) -> "torch.Tensor":
            return torch.tanh(self.base_distribution.sample())

        def rsample(self) -> "torch.Tensor":
            return torch.tanh(self.base_distribution.rsample())

        def log_prob(self, action: "torch.Tensor") -> "torch.Tensor":
            epsilon = torch.finfo(action.dtype).eps
            bounded_action = action.clamp(-1.0 + epsilon, 1.0 - epsilon)
            raw_action = torch.atanh(bounded_action)
            log_det_jacobian = torch.log1p(-bounded_action.pow(2))
            return self.base_distribution.log_prob(raw_action) - log_det_jacobian

        def entropy(self) -> "torch.Tensor":
            # A tanh-transformed Normal has no closed-form entropy. One
            # reparameterized sample provides a differentiable Monte Carlo estimate.
            action = self.rsample()
            return -self.log_prob(action)

    class ActorCritic(nn.Module):
        def __init__(self, observation_dim: int, num_actions: int, hidden_sizes: Sequence[int]) -> None:
            super().__init__()
            layers = []
            last_dim = observation_dim
            for hidden_size in hidden_sizes:
                layers.extend([nn.Linear(last_dim, hidden_size), nn.Tanh()])
                last_dim = hidden_size
            self.shared = nn.Sequential(*layers)
            self.mean_head = nn.Linear(last_dim, num_actions) #desired_force_mean, joint5_mean
            self.log_std_head = nn.Linear(last_dim, num_actions)
            self.value_head = nn.Linear(last_dim, 1)

        def forward(self, obs: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            features = self.shared(obs)
            return self.mean_head(features), self.log_std_head(features), self.value_head(features).squeeze(-1)

        def distribution(self, obs: "torch.Tensor") -> "SquashedNormal":
            mean, log_std, _ = self(obs)
            log_std = torch.clamp(log_std, -5, 2)
            std = torch.exp(log_std)
            return SquashedNormal(mean, std)

        def value(self, obs: "torch.Tensor") -> "torch.Tensor":
            _, _, value = self(obs)
            return value

else:

    class ActorCritic:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


class RolloutBuffer:
    def __init__(
        self,
        observation_dim: int,
        num_actions: int,
        size: int,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        self.obs_buf = np.zeros((size, observation_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, num_actions), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
    ) -> None:
        if self.ptr >= self.max_size:
            raise RuntimeError("RolloutBuffer is full; call get() before storing more samples.")
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = action
        self.rew_buf[self.ptr] = float(reward)
        self.val_buf[self.ptr] = float(value)
        self.logp_buf[self.ptr] = float(log_prob)
        self.ptr += 1

    def finish_path(self, last_value: float = 0.0) -> None:
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rew_buf[path_slice], last_value)
        values = np.append(self.val_buf[path_slice], last_value)
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.adv_buf[path_slice] = _discount_cumsum(deltas, self.gamma * self.gae_lambda)
        self.ret_buf[path_slice] = _discount_cumsum(rewards, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self, device: str) -> Dict[str, "torch.Tensor"]:
        _require_torch()
        if self.ptr != self.max_size:
            raise RuntimeError("RolloutBuffer must be full before get().")
        self.ptr = 0
        self.path_start_idx = 0

        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf) + 1e-8
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        return {
            "obs": torch.as_tensor(self.obs_buf, dtype=torch.float32, device=device),
            "act": torch.as_tensor(self.act_buf, dtype=torch.float32, device=device),
            "ret": torch.as_tensor(self.ret_buf, dtype=torch.float32, device=device),
            "adv": torch.as_tensor(self.adv_buf, dtype=torch.float32, device=device),
            "logp": torch.as_tensor(self.logp_buf, dtype=torch.float32, device=device),
        }


class PPOAgent:
    def __init__(self, config: PPOConfig) -> None:
        _require_torch()
        self.config = config
        self.device = torch.device(config.device)
        self.model = ActorCritic(
            observation_dim=config.observation_dim,
            num_actions=config.num_actions,
            hidden_sizes=config.hidden_sizes,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))
        self.environment_metadata: Optional[Dict[str, Any]] = None

    def step(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        actions, log_probs, values = self.step_batch(np.asarray(obs)[None, :])
        return actions[0], float(log_probs[0]), float(values[0])

    def step_batch(
        self,
        observations: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim != 2:
            raise ValueError(
                "Batched observations must have shape "
                f"(num_envs, observation_dim), got {observations.shape}"
            )
        obs_tensor = torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            distribution = self.model.distribution(obs_tensor)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).sum(-1)
            value = self.model.value(obs_tensor)
        return (
            action.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().numpy(),
        )

    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            distribution = self.model.distribution(obs_tensor)
            if deterministic:
                action = distribution.mean
            else:
                action = distribution.sample()
        return action.squeeze(0).cpu().numpy()

    def value(self, obs: np.ndarray) -> float:
        return float(self.value_batch(np.asarray(obs)[None, :])[0])

    def value_batch(self, observations: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim != 2:
            raise ValueError(
                "Batched observations must have shape "
                f"(num_envs, observation_dim), got {observations.shape}"
            )
        obs_tensor = torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            value = self.model.value(obs_tensor)
        return value.cpu().numpy()

    def update(self, data: Dict[str, "torch.Tensor"]) -> Dict[str, float]:
        obs = data["obs"]
        act = data["act"]
        ret = data["ret"]
        adv = data["adv"]
        old_logp = data["logp"]
        num_samples = obs.shape[0]

        metrics: Dict[str, float] = {}
        for train_iter in range(self.config.train_iters):
            permutation = torch.randperm(num_samples, device=self.device)
            kl_values = []
            entropy_values = []
            policy_losses = []
            value_losses = []

            for start in range(0, num_samples, self.config.batch_size):
                idx = permutation[start : start + self.config.batch_size]
                distribution = self.model.distribution(obs[idx])
                logp = distribution.log_prob(act[idx]).sum(-1)
                value = self.model.value(obs[idx])
                ratio = torch.exp(logp - old_logp[idx])

                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                )
                policy_loss = -torch.min(ratio * adv[idx], clipped_ratio * adv[idx]).mean()
                value_loss = ((value - ret[idx]) ** 2).mean()
                entropy = distribution.entropy().sum(-1).mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                if self.config.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_logp[idx] - logp).mean()
                kl_values.append(float(approx_kl.item()))
                entropy_values.append(float(entropy.item()))
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))

            metrics = {
                "train_iters": float(train_iter + 1),
                "kl": float(np.mean(kl_values)),
                "entropy": float(np.mean(entropy_values)),
                "policy_loss": float(np.mean(policy_losses)),
                "value_loss": float(np.mean(value_losses)),
            }
            if metrics["kl"] > 1.5 * self.config.target_kl:
                break

        return metrics

    def save(self, path: Union[str, Path]) -> None:
        _require_torch()
        torch.save(
            {
                "config": self.config,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "environment_metadata": self.environment_metadata,
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None) -> "PPOAgent":
        _require_torch()
        checkpoint = torch.load(Path(path), map_location=device or "cpu")
        config = checkpoint["config"]
        if device is not None:
            config.device = device
        agent = cls(config)
        agent.model.load_state_dict(checkpoint["model_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.environment_metadata = checkpoint.get("environment_metadata")
        return agent


def _subprocess_env_worker(connection: Any, env_factory: Callable[[], Any]) -> None:
    env = None
    try:
        env = env_factory()
        connection.send(
            (
                "ready",
                {
                    "observation_dim": int(env.observation_dim),
                    "num_actions": int(env.num_actions),
                },
            )
        )
        while True:
            command, payload = connection.recv()
            if command == "reset":
                connection.send(("ok", env.reset()))
            elif command == "step":
                connection.send(("ok", env.step(payload)))
            elif command == "close":
                close = getattr(env, "close", None)
                if callable(close):
                    close()
                connection.send(("ok", None))
                return
            else:
                raise ValueError(f"Unknown environment worker command: {command!r}")
    except EOFError:
        return
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        connection.close()


class SubprocessVectorEnv:
    """Run independent environments in spawned worker processes."""

    def __init__(self, env_factories: Sequence[Callable[[], Any]]) -> None:
        if not env_factories:
            raise ValueError("At least one environment factory is required")

        self._closed = False
        self._connections: List[Any] = []
        self._processes: List[mp.Process] = []
        context = mp.get_context("spawn")

        try:
            for env_factory in env_factories:
                parent_connection, child_connection = context.Pipe()
                process = context.Process(
                    target=_subprocess_env_worker,
                    args=(child_connection, env_factory),
                    daemon=True,
                )
                process.start()
                child_connection.close()
                self._connections.append(parent_connection)
                self._processes.append(process)

            environment_specs = [
                self._receive(index, expected_status="ready")
                for index in range(len(self._connections))
            ]
            first_spec = environment_specs[0]
            for worker_index, spec in enumerate(environment_specs[1:], start=1):
                if spec != first_spec:
                    raise ValueError(
                        f"Environment worker {worker_index} has incompatible dimensions: "
                        f"{spec!r} != {first_spec!r}"
                    )
            self.observation_dim = int(first_spec["observation_dim"])
            self.num_actions = int(first_spec["num_actions"])
        except BaseException:
            self.close()
            raise

    @property
    def num_envs(self) -> int:
        return len(self._connections)

    def reset(self, worker_indices: Optional[Sequence[int]] = None) -> np.ndarray:
        indices = self._resolve_indices(worker_indices)
        for index in indices:
            self._connections[index].send(("reset", None))
        observations = [self._receive(index) for index in indices]
        return np.asarray(observations, dtype=np.float32)

    def step(
        self,
        worker_indices: Sequence[int],
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        indices = self._resolve_indices(worker_indices)
        actions = np.asarray(actions, dtype=np.float32)
        expected_shape = (len(indices), self.num_actions)
        if actions.shape != expected_shape:
            raise ValueError(
                f"Expected batched action shape {expected_shape}, got {actions.shape}"
            )

        for index, action in zip(indices, actions):
            self._connections[index].send(("step", action))
        results = [self._receive(index) for index in indices]
        observations, rewards, dones, infos = zip(*results)
        return (
            np.asarray(observations, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
            list(infos),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        for connection, process in zip(self._connections, self._processes):
            if process.is_alive():
                try:
                    connection.send(("close", None))
                except (BrokenPipeError, EOFError):
                    pass
        for index, (connection, process) in enumerate(
            zip(self._connections, self._processes)
        ):
            if process.is_alive():
                try:
                    self._receive(index)
                except (RuntimeError, EOFError, BrokenPipeError):
                    pass
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
            connection.close()

    def _resolve_indices(
        self,
        worker_indices: Optional[Sequence[int]],
    ) -> List[int]:
        indices = (
            list(range(self.num_envs))
            if worker_indices is None
            else [int(index) for index in worker_indices]
        )
        if len(set(indices)) != len(indices):
            raise ValueError("Environment worker indices must be unique")
        for index in indices:
            if index < 0 or index >= self.num_envs:
                raise IndexError(f"Environment worker index {index} is out of range")
        return indices

    def _receive(self, worker_index: int, expected_status: str = "ok") -> Any:
        status, payload = self._connections[worker_index].recv()
        if status == "error":
            raise RuntimeError(
                f"Environment worker {worker_index} failed:\n{payload}"
            )
        if status != expected_status:
            raise RuntimeError(
                f"Environment worker {worker_index} returned unexpected status "
                f"{status!r}; expected {expected_status!r}"
            )
        return payload


class PPOTrainer:
    def __init__(self, env: Any, agent: PPOAgent, config: Optional[PPOConfig] = None) -> None:
        self.env = env
        self.agent = agent
        self.config = config or agent.config
        self.completed_episodes = 0

    def collect_rollout(self) -> Tuple[Dict[str, "torch.Tensor"], Dict[str, Any]]:
        buffer = RolloutBuffer(
            observation_dim=self.config.observation_dim,
            num_actions=self.env.num_actions,
            size=self.config.rollout_steps,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        obs = self.env.reset()
        episode_return = 0.0
        episode_length = 0
        completed_returns = []
        completed_lengths = []
        completed_efficiency_rewards = []
        completed_mass_metrics: List[Dict[str, float]] = []
        successful_episode_forces = []
        recent_gripper_forces = deque(maxlen=5)
        episode_desired_force_sum = 0.0
        episode_measured_force_sum = 0.0
        episode_force_samples = 0
        episode_slipped = False
        successes = 0
        completed_episodes = 0
        efficiency_confirmations = 0

        for step_idx in range(self.config.rollout_steps):
            action, log_prob, value = self.agent.step(obs)
            next_obs, reward, done, info = self.env.step(action)
            buffer.store(obs, action, reward, value, log_prob)

            average_gripper_force = 0.5 * (
                float(info["left_force_n"]) + float(info["right_force_n"])
            )
            recent_gripper_forces.append(average_gripper_force)
            episode_desired_force_sum += float(info.get("desired_force_n", 0.0))
            episode_measured_force_sum += average_gripper_force
            episode_force_samples += 1
            episode_slipped = episode_slipped or bool(
                info.get("is_slipping", False)
            )
            if float(info.get("force_efficiency_reward", 0.0)) > 0.0:
                efficiency_confirmations += 1

            episode_return += reward
            episode_length += 1
            obs = next_obs

            timeout = step_idx == self.config.rollout_steps - 1
            if done or timeout:
                terminal = done and not bool(info.get("truncated", False))
                last_value = 0.0 if terminal else self.agent.value(obs)
                buffer.finish_path(last_value)
                if done:
                    self.completed_episodes += 1
                    completed_episodes += 1
                    completed_returns.append(episode_return)
                    completed_lengths.append(episode_length)
                    completed_efficiency_rewards.append(
                        float(info.get("episode_force_efficiency_reward", 0.0))
                    )
                    success = bool(info.get("success", False))
                    successes += int(success)
                    if (
                        "box_mass_kg" in info
                        and episode_force_samples > 0
                    ):
                        completed_mass_metrics.append(
                            {
                                "box_mass_kg": float(info["box_mass_kg"]),
                                "success": float(success),
                                "slipped": float(episode_slipped),
                                "mean_desired_force_n": (
                                    episode_desired_force_sum
                                    / episode_force_samples
                                ),
                                "mean_measured_force_n": (
                                    episode_measured_force_sum
                                    / episode_force_samples
                                ),
                                "efficiency_reward": float(
                                    info.get(
                                        "episode_force_efficiency_reward",
                                        0.0,
                                    )
                                ),
                            }
                        )
                    if success:
                        successful_episode_forces.append(float(np.mean(recent_gripper_forces)))
                    print(
                        f"episode={self.completed_episodes} "
                        f"termination_reason={info.get('reason') or 'unknown'}"
                    )
                    obs = self.env.reset()
                    episode_return = 0.0
                    episode_length = 0
                    recent_gripper_forces.clear()
                    episode_desired_force_sum = 0.0
                    episode_measured_force_sum = 0.0
                    episode_force_samples = 0
                    episode_slipped = False

        rollout_info = {
            "episodes": float(completed_episodes),
            "success_rate": float(successes / completed_episodes) if completed_episodes else 0.0,
            "mean_return": float(np.mean(completed_returns)) if completed_returns else 0.0,
            "mean_length": float(np.mean(completed_lengths)) if completed_lengths else 0.0,
            "mean_success_gripper_force": (
                float(np.mean(successful_episode_forces)) if successful_episode_forces else 0.0
            ),
            "mean_efficiency_reward": (
                float(np.mean(completed_efficiency_rewards))
                if completed_efficiency_rewards
                else 0.0
            ),
            "efficiency_confirmations": float(efficiency_confirmations),
            "mass_metrics": _summarize_mass_metrics(completed_mass_metrics),
        }
        return buffer.get(self.config.device), rollout_info

    def train(self, total_timesteps: int, save_path: Optional[Union[str, Path]] = None) -> None:
        updates = max(1, int(total_timesteps) // self.config.rollout_steps)
        for update_idx in range(updates):
            data, rollout_info = self.collect_rollout()
            update_info = self.agent.update(data)
            print(
                "update={update} episodes={episodes:.0f} success_rate={success_rate:.3f} "
                "mean_return={mean_return:.3f} mean_length={mean_length:.1f} "
                "mean_success_gripper_force={mean_success_gripper_force:.3f} "
                "mean_efficiency_reward={mean_efficiency_reward:.3f} "
                "efficiency_confirmations={efficiency_confirmations:.0f} "
                "kl={kl:.5f} entropy={entropy:.3f}".format(
                    update=update_idx + 1,
                    **rollout_info,
                    **update_info,
                )
            )
            _print_mass_metrics(rollout_info["mass_metrics"])
            if save_path is not None:
                self.agent.save(save_path)


class ParallelPPOTrainer:
    """Collect PPO rollouts concurrently from subprocess environments."""

    def __init__(
        self,
        env: SubprocessVectorEnv,
        agent: PPOAgent,
        config: Optional[PPOConfig] = None,
    ) -> None:
        self.env = env
        self.agent = agent
        self.config = config or agent.config
        self.completed_episodes = 0
        if self.env.num_envs > self.config.rollout_steps:
            raise ValueError(
                "num_envs cannot exceed rollout_steps because every worker must "
                "contribute at least one sample"
            )

    def collect_rollout(self) -> Tuple[Dict[str, "torch.Tensor"], Dict[str, Any]]:
        buffer = RolloutBuffer(
            observation_dim=self.config.observation_dim,
            num_actions=self.env.num_actions,
            size=self.config.rollout_steps,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        observations = self.env.reset()
        base_steps, extra_steps = divmod(
            self.config.rollout_steps,
            self.env.num_envs,
        )
        remaining_steps = [
            base_steps + int(index < extra_steps)
            for index in range(self.env.num_envs)
        ]
        pending_paths: List[List[Tuple[np.ndarray, np.ndarray, float, float, float]]] = [
            [] for _ in range(self.env.num_envs)
        ]
        episode_returns = [0.0 for _ in range(self.env.num_envs)]
        episode_lengths = [0 for _ in range(self.env.num_envs)]
        recent_gripper_forces = [
            deque(maxlen=5) for _ in range(self.env.num_envs)
        ]
        episode_desired_force_sums = [0.0 for _ in range(self.env.num_envs)]
        episode_measured_force_sums = [0.0 for _ in range(self.env.num_envs)]
        episode_force_samples = [0 for _ in range(self.env.num_envs)]
        episode_slipped = [False for _ in range(self.env.num_envs)]

        completed_returns: List[float] = []
        completed_lengths: List[int] = []
        completed_efficiency_rewards: List[float] = []
        completed_mass_metrics: List[Dict[str, float]] = []
        successful_episode_forces: List[float] = []
        successes = 0
        completed_episodes = 0
        efficiency_confirmations = 0

        while any(steps > 0 for steps in remaining_steps):
            active_indices = [
                index
                for index, steps in enumerate(remaining_steps)
                if steps > 0
            ]
            active_observations = observations[active_indices]
            actions, log_probs, values = self.agent.step_batch(active_observations)
            next_observations, rewards, dones, infos = self.env.step(
                active_indices,
                actions,
            )
            reset_indices: List[int] = []

            for batch_index, worker_index in enumerate(active_indices):
                info = infos[batch_index]
                pending_paths[worker_index].append(
                    (
                        observations[worker_index].copy(),
                        actions[batch_index].copy(),
                        float(rewards[batch_index]),
                        float(values[batch_index]),
                        float(log_probs[batch_index]),
                    )
                )
                observations[worker_index] = next_observations[batch_index]
                remaining_steps[worker_index] -= 1

                average_gripper_force = 0.5 * (
                    float(info["left_force_n"]) + float(info["right_force_n"])
                )
                recent_gripper_forces[worker_index].append(average_gripper_force)
                episode_desired_force_sums[worker_index] += float(
                    info.get("desired_force_n", 0.0)
                )
                episode_measured_force_sums[worker_index] += average_gripper_force
                episode_force_samples[worker_index] += 1
                episode_slipped[worker_index] = (
                    episode_slipped[worker_index]
                    or bool(info.get("is_slipping", False))
                )
                if float(info.get("force_efficiency_reward", 0.0)) > 0.0:
                    efficiency_confirmations += 1
                episode_returns[worker_index] += float(rewards[batch_index])
                episode_lengths[worker_index] += 1

                if not bool(dones[batch_index]):
                    continue

                terminal = not bool(info.get("truncated", False))
                last_value = (
                    0.0
                    if terminal
                    else self.agent.value(observations[worker_index])
                )
                self._finish_pending_path(
                    buffer,
                    pending_paths[worker_index],
                    last_value,
                )

                self.completed_episodes += 1
                completed_episodes += 1
                completed_returns.append(episode_returns[worker_index])
                completed_lengths.append(episode_lengths[worker_index])
                completed_efficiency_rewards.append(
                    float(info.get("episode_force_efficiency_reward", 0.0))
                )
                success = bool(info.get("success", False))
                successes += int(success)
                if (
                    "box_mass_kg" in info
                    and episode_force_samples[worker_index] > 0
                ):
                    completed_mass_metrics.append(
                        {
                            "box_mass_kg": float(info["box_mass_kg"]),
                            "success": float(success),
                            "slipped": float(episode_slipped[worker_index]),
                            "mean_desired_force_n": (
                                episode_desired_force_sums[worker_index]
                                / episode_force_samples[worker_index]
                            ),
                            "mean_measured_force_n": (
                                episode_measured_force_sums[worker_index]
                                / episode_force_samples[worker_index]
                            ),
                            "efficiency_reward": float(
                                info.get(
                                    "episode_force_efficiency_reward",
                                    0.0,
                                )
                            ),
                        }
                    )
                if success:
                    successful_episode_forces.append(
                        float(np.mean(recent_gripper_forces[worker_index]))
                    )
                print(
                    f"episode={self.completed_episodes} worker={worker_index} "
                    f"termination_reason={info.get('reason') or 'unknown'}"
                )

                episode_returns[worker_index] = 0.0
                episode_lengths[worker_index] = 0
                recent_gripper_forces[worker_index].clear()
                episode_desired_force_sums[worker_index] = 0.0
                episode_measured_force_sums[worker_index] = 0.0
                episode_force_samples[worker_index] = 0
                episode_slipped[worker_index] = False
                if remaining_steps[worker_index] > 0:
                    reset_indices.append(worker_index)

            if reset_indices:
                reset_observations = self.env.reset(reset_indices)
                for reset_index, worker_index in enumerate(reset_indices):
                    observations[worker_index] = reset_observations[reset_index]

        unfinished_indices = [
            index for index, path in enumerate(pending_paths) if path
        ]
        if unfinished_indices:
            last_values = self.agent.value_batch(observations[unfinished_indices])
            for batch_index, worker_index in enumerate(unfinished_indices):
                self._finish_pending_path(
                    buffer,
                    pending_paths[worker_index],
                    float(last_values[batch_index]),
                )

        rollout_info = {
            "episodes": float(completed_episodes),
            "success_rate": (
                float(successes / completed_episodes) if completed_episodes else 0.0
            ),
            "mean_return": (
                float(np.mean(completed_returns)) if completed_returns else 0.0
            ),
            "mean_length": (
                float(np.mean(completed_lengths)) if completed_lengths else 0.0
            ),
            "mean_success_gripper_force": (
                float(np.mean(successful_episode_forces))
                if successful_episode_forces
                else 0.0
            ),
            "mean_efficiency_reward": (
                float(np.mean(completed_efficiency_rewards))
                if completed_efficiency_rewards
                else 0.0
            ),
            "efficiency_confirmations": float(efficiency_confirmations),
            "mass_metrics": _summarize_mass_metrics(completed_mass_metrics),
        }
        return buffer.get(self.config.device), rollout_info

    @staticmethod
    def _finish_pending_path(
        buffer: RolloutBuffer,
        path: List[Tuple[np.ndarray, np.ndarray, float, float, float]],
        last_value: float,
    ) -> None:
        for observation, action, reward, value, log_prob in path:
            buffer.store(observation, action, reward, value, log_prob)
        buffer.finish_path(last_value)
        path.clear()

    def train(
        self,
        total_timesteps: int,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        updates = max(1, int(total_timesteps) // self.config.rollout_steps)
        for update_idx in range(updates):
            data, rollout_info = self.collect_rollout()
            update_info = self.agent.update(data)
            print(
                "update={update} episodes={episodes:.0f} success_rate={success_rate:.3f} "
                "mean_return={mean_return:.3f} mean_length={mean_length:.1f} "
                "mean_success_gripper_force={mean_success_gripper_force:.3f} "
                "mean_efficiency_reward={mean_efficiency_reward:.3f} "
                "efficiency_confirmations={efficiency_confirmations:.0f} "
                "kl={kl:.5f} entropy={entropy:.3f}".format(
                    update=update_idx + 1,
                    **rollout_info,
                    **update_info,
                )
            )
            _print_mass_metrics(rollout_info["mass_metrics"])
            if save_path is not None:
                self.agent.save(save_path)
