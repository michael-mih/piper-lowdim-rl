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
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-iters", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device (auto, cpu, cuda, or cuda:N). Defaults to PPOConfig.device.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-force", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
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
    elif device is not None and device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested PyTorch device {device!r}, but CUDA is not available."
            )

    controller = SimController(
        pid_controllers=[PIDController(0.01, 0.0, 0.0) for _ in range(8)],
        model_path=args.model_path,
        render=not args.no_render,
    )

    env_config_kwargs = {}
    if args.max_steps is not None:
        env_config_kwargs["max_steps"] = args.max_steps

    observation_config_kwargs = {}
    if args.min_force is not None:
        observation_config_kwargs["min_force_threshold_n"] = args.min_force

    env = GraspPPOEnv(
        controller=controller,
        env_config=GraspEnvConfig(**env_config_kwargs),
        observation_config=ObservationConfig(**observation_config_kwargs),
        reward_config=RewardConfig(),
        seed=args.seed,
    )

    ppo_config_kwargs = {}
    if args.learning_rate is not None:
        ppo_config_kwargs["learning_rate"] = args.learning_rate
    if args.rollout_steps is not None:
        ppo_config_kwargs["rollout_steps"] = args.rollout_steps
    if args.batch_size is not None:
        ppo_config_kwargs["batch_size"] = args.batch_size
    if args.train_iters is not None:
        ppo_config_kwargs["train_iters"] = args.train_iters
    if device is not None:
        ppo_config_kwargs["device"] = device

    config = PPOConfig(
        observation_dim=env.observation_dim,
        num_actions=env.num_actions,
        **ppo_config_kwargs,
    )
    total_timesteps = args.total_timesteps if args.total_timesteps is not None else 60000
    print(f"training_device={config.device}")
    agent = PPOAgent(config)
    trainer = PPOTrainer(env, agent, config)
    trainer.train(total_timesteps=total_timesteps, save_path=Path(args.save_path))


if __name__ == "__main__":
    main()
