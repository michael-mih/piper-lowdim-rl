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
    parser = argparse.ArgumentParser(description="Run a trained PPO gripper policy in MuJoCo.")
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
        default="sim"
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

    c = args.controller 
    if c == "sim":
        controller = SimController(
            pid_controllers=[PIDController(0.01, 0.0, 0.0) for _ in range(8)],
            model_path=args.model_path,
            render=not args.no_render,
        )

    else:
        controller = PhysController(
            pid_controllers=[PIDController(0.01, 0.0, 0.0) for _ in range(8)]
        )
    env_config_kwargs = {}
    if args.max_steps is not None:
        env_config_kwargs["max_steps"] = args.max_steps

    observation_config_kwargs = {}
    if args.min_force is not None:
        observation_config_kwargs["min_force_threshold_n"] = args.min_force

    env = GraspPPOEnv(
        controller=controller,
        object_height_fn= lambda controller: None,
        env_config=GraspEnvConfig(**env_config_kwargs),
        observation_config=ObservationConfig(**observation_config_kwargs),
        reward_config=RewardConfig(),
        seed=args.seed,
    )
    agent = PPOAgent.load(args.checkpoint, device=device)
    agent.model.eval()

    if agent.config.observation_dim != env.observation_dim:
        raise ValueError(
            f"Checkpoint expects {agent.config.observation_dim} observations, "
            f"but the environment provides {env.observation_dim}."
        )
    if agent.config.num_actions != env.num_actions:
        raise ValueError(
            f"Checkpoint expects {agent.config.num_actions} actions, "
            f"but the environment provides {env.num_actions}."
        )



    observation = env.reset()
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
    if not args.no_render:
        while True:
            env.controller.step()


    

    


if __name__ == "__main__":
    main()
