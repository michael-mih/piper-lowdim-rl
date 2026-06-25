from __future__ import annotations

import argparse

import numpy as np

from controllers.sim_controller import SimController
from learning.env import GraspAction, GraspEnvConfig, GraspPPOEnv, ObservationConfig, RewardConfig
from learning.ppo import PPOAgent
from scripts.build_sim import combined_xml
from scripts.train_ppo_grasp import PIDController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained PPO gripper policy in MuJoCo.")
    parser.add_argument("checkpoint", help="Checkpoint produced by train-ppo-grasp.")
    parser.add_argument("--model-path", default=str(combined_xml), help="MuJoCo XML path.")

    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--min-force", type=float, default=1.0)
    parser.add_argument("--device", default="cpu", help="PyTorch device, for example cpu or cuda.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
    agent = PPOAgent.load(args.checkpoint, device=args.device)
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
                f"action={GraspAction(action).name} reward={reward:.4f}"
            )
    print(f"termination_reason={info.get('reason') or 'unknown'}")
    if not args.no_render:
        while True:
            env.controller.step()


    

    


if __name__ == "__main__":
    main()
