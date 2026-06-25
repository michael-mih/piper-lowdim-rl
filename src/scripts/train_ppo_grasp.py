from __future__ import annotations

import argparse
from pathlib import Path

from controllers.sim_controller import SimController
from learning.env import GraspEnvConfig, GraspPPOEnv, ObservationConfig, RewardConfig
from learning.ppo import PPOAgent, PPOConfig, PPOTrainer
from scripts.build_sim import combined_xml


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, dt: float = 0.01) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral_error = 0.0
        self.prev_error = 0.0

    def calculate(self, target: float, current: float) -> float:
        error = target - current
        self.integral_error += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral_error + self.kd * derivative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight PPO-clip gripper policy.")
    parser.add_argument("--model-path", default=str(combined_xml), help="MuJoCo XML path.")
    parser.add_argument("--save-path", default="ppo_grasp_policy.pt", help="Where to write checkpoints.")
    parser.add_argument("--total-timesteps", type=int, default=6000)
    parser.add_argument("--rollout-steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-iters", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device (auto, cpu, cuda, or cuda:N).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-force", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable the MuJoCo viewer and run simulation as fast as possible.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested PyTorch device {device!r}, but CUDA is not available."
            )

    print(f"training_device={device}")
    controller = SimController(
        pid_controllers=[PIDController(0.01, 0.0, 0.0) for _ in range(8)],
        model_path=args.model_path,
        render=not args.no_render,
    )
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
    trainer = PPOTrainer(env, agent, config)
    trainer.train(total_timesteps=args.total_timesteps, save_path=Path(args.save_path))


if __name__ == "__main__":
    main()
