from __future__ import annotations

import argparse

import numpy as np

from controllers.sim_controller import SimController
from controllers.phys_controller import PhysController
from learning.env import GraspEnvConfig, GraspPPOEnv, ObservationConfig, RewardConfig
from learning.ppo import PPOAgent
from scripts.build_sim import combined_xml
from scripts.train_ppo_grasp import PIDController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained PPO gripper policy in simulation or on Piper."
    )
    parser.add_argument("checkpoint", help="Checkpoint produced by train-ppo-grasp.")
    parser.add_argument("--model-path", default=str(combined_xml), help="MuJoCo XML path.")

    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--min-force", type=float, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device (auto, cpu, cuda, or cuda:N). Defaults to checkpoint PPOConfig.device.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable the MuJoCo viewer and run simulation as fast as possible.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from the policy instead of selecting its highest-probability action.",
    )
    parser.add_argument(
        "--controller",
        choices=("sim", "phys"),
        default="sim",
    )
    parser.add_argument("--can-channel", default="can0")
    parser.add_argument("--speed-percent", type=int, default=50)
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

    env_config_kwargs = {}
    if args.max_steps is not None:
        env_config_kwargs["max_steps"] = args.max_steps

    observation_config_kwargs = {}
    if args.min_force is not None:
        observation_config_kwargs["min_force_threshold_n"] = args.min_force

    agent = PPOAgent.load(args.checkpoint, device=device)
    agent.model.eval()

    expected_observation_dim = len(GraspPPOEnv.observation_names)
    if agent.config.observation_dim != expected_observation_dim:
        raise ValueError(
            f"Checkpoint expects {agent.config.observation_dim} observations, "
            f"but the environment provides {expected_observation_dim}."
        )
    if agent.config.num_actions != GraspPPOEnv.num_actions:
        raise ValueError(
            f"Checkpoint expects {agent.config.num_actions} actions, "
            f"but the environment provides {GraspPPOEnv.num_actions}."
        )

    controller = None
    try:
        if args.controller == "sim":
            controller = SimController(
                pid_controllers=[
                    PIDController(0.01, 0.0, 0.0) for _ in range(8)
                ],
                model_path=args.model_path,
                render=not args.no_render,
            )
        else:
            controller = PhysController(
                pid_controllers=[
                    PIDController(0.01, 0.0, 0.0) for _ in range(8)
                ],
                channel=args.can_channel,
                speed_percent=args.speed_percent,
            )

        env = GraspPPOEnv(
            controller=controller,
            object_height_fn=lambda controller: None,
            env_config=GraspEnvConfig(**env_config_kwargs),
            observation_config=ObservationConfig(**observation_config_kwargs),
            reward_config=RewardConfig(),
            seed=args.seed,
        )

        observation = env.reset()
        calibration = env.last_info.get("calibration", {})
        if args.controller == "phys" and not calibration.get(
            "reached_threshold", False
        ):
            raise RuntimeError(
                "Physical grasp calibration ended without reaching the force threshold"
            )

        done = False
        info = {}
        while not done:
            action = agent.act(
                observation,
                deterministic=not args.stochastic,
            )
            observation, reward, done, info = env.step(action)
            if not done:
                print(
                    f"step={env.steps} "
                    f"action={np.asarray(action).tolist()} reward={reward:.4f}"
                )
        print(f"termination_reason={info.get('reason') or 'unknown'}")

        if args.controller == "sim" and not args.no_render:
            while True:
                env.controller.step()
    finally:
        if controller is not None and hasattr(controller, "stop"):
            controller.stop()


if __name__ == "__main__":
    main()
