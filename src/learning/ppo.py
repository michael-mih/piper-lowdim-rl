from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

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
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            distribution = self.model.distribution(obs_tensor)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).sum(-1)
            value = self.model.value(obs_tensor)
        return action.squeeze(0).cpu().numpy(), float(log_prob.item()), float(value.item())

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
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            value = self.model.value(obs_tensor)
        return float(value.item())

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


class PPOTrainer:
    def __init__(self, env: Any, agent: PPOAgent, config: Optional[PPOConfig] = None) -> None:
        self.env = env
        self.agent = agent
        self.config = config or agent.config
        self.completed_episodes = 0

    def collect_rollout(self) -> Tuple[Dict[str, "torch.Tensor"], Dict[str, float]]:
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
        successful_episode_forces = []
        recent_gripper_forces = deque(maxlen=5)
        successes = 0
        completed_episodes = 0

        for step_idx in range(self.config.rollout_steps):
            action, log_prob, value = self.agent.step(obs)
            next_obs, reward, done, info = self.env.step(action)
            buffer.store(obs, action, reward, value, log_prob)

            average_gripper_force = 0.5 * (
                float(info["left_force_n"]) + float(info["right_force_n"])
            )
            recent_gripper_forces.append(average_gripper_force)

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
                    success = bool(info.get("success", False))
                    successes += int(success)
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

        rollout_info = {
            "episodes": float(completed_episodes),
            "success_rate": float(successes / completed_episodes) if completed_episodes else 0.0,
            "mean_return": float(np.mean(completed_returns)) if completed_returns else 0.0,
            "mean_length": float(np.mean(completed_lengths)) if completed_lengths else 0.0,
            "mean_success_gripper_force": (
                float(np.mean(successful_episode_forces)) if successful_episode_forces else 0.0
            ),
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
                "kl={kl:.5f} entropy={entropy:.3f}".format(
                    update=update_idx + 1,
                    **rollout_info,
                    **update_info,
                )
            )
            if save_path is not None:
                self.agent.save(save_path)
