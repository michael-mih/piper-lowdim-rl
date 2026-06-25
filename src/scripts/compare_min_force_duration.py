from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from controllers.sim_controller import SimController
from fsm.custom_min_grasp_fsm import FSMActor
from learning.env import GraspEnvConfig, GraspPPOEnv, ObservationConfig, RewardConfig
from learning.ppo import PPOAgent, PPOConfig, RolloutBuffer
from scripts.build_sim import combined_xml
from scripts.train_ppo_grasp import PIDController


Sample = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PPO training and the FSM grasp test, then plot best/min force over elapsed time."
    )
    parser.add_argument("--model-path", default=str(combined_xml), help="MuJoCo XML path.")
    parser.add_argument("--plot-path", default="min_force_duration.png")
    parser.add_argument("--csv-path", default="min_force_duration.csv")
    parser.add_argument("--save-path", default="", help="Optional PPO checkpoint path. Empty disables saving.")

    parser.add_argument("--total-timesteps", type=int, default=30000)
    parser.add_argument("--rollout-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-iters", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-force", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--render-ppo", action="store_true", help="Render the PPO simulation.")

    parser.add_argument("--render-fsm", action="store_true", help="Render the FSM simulation.")
    parser.add_argument("--fsm-increment", type=float, default=0.00001)
    parser.add_argument("--fsm-convergence-interval", type=float, default=10.0)
    parser.add_argument("--fsm-sleep-s", type=float, default=0.01)
    parser.add_argument("--fsm-max-steps", type=int, default=20000)
    parser.add_argument("--fsm-prep-max-steps", type=int, default=5000)
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested PyTorch device {device!r}, but CUDA is not available.")
    return device


def make_controller(model_path: str, render: bool) -> SimController:
    return SimController(
        pid_controllers=[PIDController(0.01, 0.0, 0.0) for _ in range(8)],
        model_path=model_path,
        render=render,
    )


def successful_episode_force(env: GraspPPOEnv, fallback_force: float) -> float:
    if env.episode_force_samples:
        return float(env.episode_force_sum / env.episode_force_samples)
    return float(fallback_force)


def collect_tracked_rollout(
    env: GraspPPOEnv,
    agent: PPOAgent,
    config: PPOConfig,
    start_time: float,
    samples: List[Sample],
    completed_episode_count: int,
    best_force: Optional[float],
) -> Tuple[Dict[str, Any], Dict[str, float], int, Optional[float]]:
    buffer = RolloutBuffer(
        observation_dim=config.observation_dim,
        size=config.rollout_steps,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    obs = env.reset()
    episode_return = 0.0
    episode_length = 0
    completed_returns = []
    completed_lengths = []
    successful_episode_forces = []
    recent_gripper_forces = deque(maxlen=5)
    successes = 0
    rollout_episodes = 0

    for step_idx in range(config.rollout_steps):
        action, log_prob, value = agent.step(obs)
        next_obs, reward, done, info = env.step(action)
        buffer.store(obs, action, reward, value, log_prob)

        average_gripper_force = 0.5 * (float(info["left_force_n"]) + float(info["right_force_n"]))
        recent_gripper_forces.append(average_gripper_force)

        episode_return += reward
        episode_length += 1
        obs = next_obs

        timeout = step_idx == config.rollout_steps - 1
        if done or timeout:
            terminal = done and not bool(info.get("truncated", False))
            last_value = 0.0 if terminal else agent.value(obs)
            buffer.finish_path(last_value)

            if done:
                completed_episode_count += 1
                rollout_episodes += 1
                completed_returns.append(episode_return)
                completed_lengths.append(episode_length)
                success = bool(info.get("success", False))
                successes += int(success)

                if success:
                    force = successful_episode_force(env, average_gripper_force)
                    successful_episode_forces.append(force)
                    best_force = force if best_force is None else min(best_force, force)
                    samples.append((time.perf_counter() - start_time, best_force))

                print(
                    f"episode={completed_episode_count} "
                    f"termination_reason={info.get('reason') or 'unknown'}"
                )
                obs = env.reset()
                episode_return = 0.0
                episode_length = 0
                recent_gripper_forces.clear()

    rollout_info = {
        "episodes": float(rollout_episodes),
        "success_rate": float(successes / rollout_episodes) if rollout_episodes else 0.0,
        "mean_return": float(np.mean(completed_returns)) if completed_returns else 0.0,
        "mean_length": float(np.mean(completed_lengths)) if completed_lengths else 0.0,
        "mean_success_gripper_force": (
            float(np.mean(successful_episode_forces)) if successful_episode_forces else 0.0
        ),
    }
    return buffer.get(config.device), rollout_info, completed_episode_count, best_force


def run_ppo(args: argparse.Namespace) -> List[Sample]:
    device = resolve_device(args.device)
    print(f"ppo_training_device={device}")

    controller = make_controller(args.model_path, render=args.render_ppo)
    env = GraspPPOEnv(
        controller=controller,
        env_config=GraspEnvConfig(max_steps=args.max_steps),
        observation_config=ObservationConfig(min_force_threshold_n=args.min_force),
        reward_config=RewardConfig(),
        seed=args.seed,
    )
    config = PPOConfig(
        observation_dim=env.observation_dim,
        num_actions=env.num_actions,
        learning_rate=args.learning_rate,
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        train_iters=args.train_iters,
        device=device,
    )
    agent = PPOAgent(config)

    updates = max(1, int(args.total_timesteps) // args.rollout_steps)
    completed_episode_count = 0
    best_force: Optional[float] = None
    samples: List[Sample] = []
    start_time = time.perf_counter()

    for update_idx in range(updates):
        data, rollout_info, completed_episode_count, best_force = collect_tracked_rollout(
            env=env,
            agent=agent,
            config=config,
            start_time=start_time,
            samples=samples,
            completed_episode_count=completed_episode_count,
            best_force=best_force,
        )
        update_info = agent.update(data)
        best_force_text = "nan" if best_force is None else f"{best_force:.3f}"
        print(
            "ppo_update={update} episodes={episodes:.0f} success_rate={success_rate:.3f} "
            "mean_return={mean_return:.3f} mean_length={mean_length:.1f} "
            "mean_success_gripper_force={mean_success_gripper_force:.3f} "
            "best_success_force={best_force} kl={kl:.5f} entropy={entropy:.3f}".format(
                update=update_idx + 1,
                best_force=best_force_text,
                **rollout_info,
                **update_info,
            )
        )
        if args.save_path:
            agent.save(Path(args.save_path))

    return samples


def maybe_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def viewer_closed(controller: SimController) -> bool:
    viewer = getattr(controller, "viewer", None)
    if viewer is None:
        return False
    import glfw

    return bool(glfw.window_should_close(viewer.window))


def run_fsm(args: argparse.Namespace) -> List[Sample]:
    controller = make_controller(args.model_path, render=args.render_fsm)
    fsm_actor = FSMActor(controller=controller)
    samples: List[Sample] = []
    start_time = time.perf_counter()

    target_angles = [0, 1.5, -0.3, 0, -0.7, 0, 0.03, -0.03]
    controller.set_initial_position(target_angles)
    gripper_delta = 0.0001
    arm_delta = 0.001

    prep_steps = 0
    while target_angles[4] > -1.2 and prep_steps < args.fsm_prep_max_steps:
        controller.send_joint_angle_cmd(target_angles)
        controller.step()

        if min(controller.get_force_left(), controller.get_force_right()) > 0.8:
            gripper_delta = 0
        if gripper_delta == 0 and target_angles[4] > -1.2:
            target_angles[4] -= arm_delta

        target_angles[6] -= gripper_delta
        target_angles[7] += gripper_delta
        prep_steps += 1

        if viewer_closed(controller):
            break
        maybe_sleep(args.fsm_sleep_s)

    if target_angles[4] > -1.2:
        print(f"fsm_prep_stopped_after={prep_steps} target_joint5={target_angles[4]:.3f}")

    for _ in range(args.fsm_max_steps):
        fsm_actor.step()
        if fsm_actor.min is not None:
            samples.append((time.perf_counter() - start_time, float(fsm_actor.min)))

        if fsm_actor.is_converged(args.fsm_convergence_interval):
            break
        if fsm_actor.is_slip():
            fsm_actor.tighten(args.fsm_increment)
        else:
            fsm_actor.loosen(args.fsm_increment)

        if viewer_closed(controller):
            break
        maybe_sleep(args.fsm_sleep_s)

    print(f"fsm_converged_min_force={fsm_actor.min}")
    return samples


def write_csv(path: Path, ppo_samples: List[Sample], fsm_samples: List[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "elapsed_s", "min_force_n"])
        for elapsed_s, force_n in ppo_samples:
            writer.writerow(["ppo", f"{elapsed_s:.6f}", f"{force_n:.6f}"])
        for elapsed_s, force_n in fsm_samples:
            writer.writerow(["fsm", f"{elapsed_s:.6f}", f"{force_n:.6f}"])


def plot_samples(path: Path, ppo_samples: List[Sample], fsm_samples: List[Sample]) -> None:
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError(
            "Plot output requires matplotlib. Activate the conda env where matplotlib is installed "
            "or install it in the current Python environment."
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False

    if ppo_samples:
        x, y = zip(*ppo_samples)
        ax.plot(x, y, label="PPO best successful lift force")
        plotted = True
    else:
        print("warning: PPO produced no successful episodes, so no PPO force curve was plotted.")

    if fsm_samples:
        x, y = zip(*fsm_samples)
        ax.plot(x, y, label="FSM minimum force")
        plotted = True
    else:
        print("warning: FSM produced no force samples, so no FSM force curve was plotted.")

    ax.set_xlabel("Elapsed wall-clock time within each run (s)")
    ax.set_ylabel("Minimum average gripper force (N)")
    ax.set_title("Minimum force found vs program duration")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No force samples recorded", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    print("running_ppo=true")
    ppo_samples = run_ppo(args)
    print("running_fsm=true")
    fsm_samples = run_fsm(args)

    csv_path = Path(args.csv_path)
    plot_path = Path(args.plot_path)
    write_csv(csv_path, ppo_samples, fsm_samples)
    plot_samples(plot_path, ppo_samples, fsm_samples)
    print(f"wrote_csv={csv_path}")
    print(f"wrote_plot={plot_path}")


if __name__ == "__main__":
    main()
