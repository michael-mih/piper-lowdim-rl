from __future__ import annotations

import argparse

import numpy as np

from learning.env import GraspEnvConfig, GraspPPOEnv, ObservationConfig, RewardConfig
from learning.ppo import PPOAgent


# Change this value to set the simulated cube mass in kilograms.
CUBE_MASS_KG = 0.2


def set_sim_cube_mass(controller, mass_kg: float) -> None:
    if mass_kg <= 0.0:
        raise ValueError("CUBE_MASS_KG must be greater than zero")

    model = controller.sim.model
    body_id = model.body_name2id("grasp_box")
    geom_id = model.geom_name2id("grasp_box_geom")
    half_extents = np.asarray(model.geom_size[geom_id, :3], dtype=np.float64)
    hx, hy, hz = half_extents

    model.body_mass[body_id] = mass_kg
    model.body_inertia[body_id, :3] = (mass_kg / 3.0) * np.asarray(
        (
            hy * hy + hz * hz,
            hx * hx + hz * hz,
            hx * hx + hy * hy,
        ),
        dtype=np.float64,
    )
    if hasattr(model, "body_subtreemass"):
        model.body_subtreemass[body_id] = mass_kg
    controller.sim.forward()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained PPO gripper policy in simulation or on Piper."
    )
    parser.add_argument("checkpoint", help="Checkpoint produced by train-ppo-grasp.")
    parser.add_argument(
        "--model-path",
        default=None,
        help="MuJoCo XML path. Only used with --controller sim.",
    )

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
    parser.add_argument("--gpr-model-path", default=None)
    parser.add_argument("--feedback-timeout", type=float, default=5.0)
    parser.add_argument("--max-feedback-age", type=float, default=0.1)
    parser.add_argument("--max-gpr-std", type=float, default=1.0)
    parser.add_argument("--hard-force-limit", type=float, default=5.0)
    parser.add_argument("--command-speed-percent", type=int, default=10)
    parser.add_argument(
        "--allow-hardware-motion",
        action="store_true",
        help="Required acknowledgement before the physical controller is initialized.",
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

    env_config_kwargs = {}
    if args.max_steps is not None:
        env_config_kwargs["max_steps"] = args.max_steps

    # Training uses observation randomization; evaluation uses measured values.
    observation_config_kwargs = {
        "sensor_offset_range": 0.0,
        "sensor_noise_std": 0.0,
        "motor_offset_range": 0.0,
    }
    if args.min_force is not None:
        observation_config_kwargs["min_force_threshold_n"] = args.min_force
    env_config = GraspEnvConfig(**env_config_kwargs)
    observation_config = ObservationConfig(**observation_config_kwargs)

    if args.controller == "phys" and not args.allow_hardware_motion:
        raise RuntimeError(
            "Physical execution requires --allow-hardware-motion after the workspace, "
            "start pose, E-stop, and sensor streams have been checked"
        )
    if args.controller == "phys" and observation_config.min_force_threshold_n < 0.01:
        raise ValueError("Physical calibration threshold must be at least 0.01 N")

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

    expected_metadata = GraspPPOEnv.policy_metadata_for_configs(
        env_config,
        observation_config,
    )
    if agent.environment_metadata is None:
        if args.controller == "phys":
            raise RuntimeError(
                "Checkpoint has no environment metadata and cannot be verified for hardware. "
                "Retrain it with the current safety and timing configuration."
            )
        print("warning: checkpoint has no environment metadata; compatibility is unverified")
    elif agent.environment_metadata != expected_metadata:
        raise RuntimeError(
            "Checkpoint environment metadata does not match the current observation/action "
            "and control-period configuration. Retrain or restore the matching configuration."
        )

    controller = None
    try:
        if args.controller == "sim":
            from controllers.sim_controller import SimController

            model_path = args.model_path
            if model_path is None:
                from scripts.build_sim import combined_xml

                model_path = str(combined_xml)
            controller = SimController(
                pid_controllers=[],
                model_path=model_path,
                render=not args.no_render,
            )
            set_sim_cube_mass(controller, CUBE_MASS_KG)
        else:
            from controllers.phys_controller import PhysController

            controller = PhysController(
                pid_controllers=[],
                channel=args.can_channel,
                gpr_model_path=args.gpr_model_path,
                feedback_timeout_s=args.feedback_timeout,
                max_feedback_age_s=args.max_feedback_age,
                max_gpr_std_n=args.max_gpr_std,
                hard_force_limit_n=args.hard_force_limit,
                command_speed_percent=args.command_speed_percent,
            )

        env = GraspPPOEnv(
            controller=controller,
            object_height_fn=lambda controller: None,
            env_config=env_config,
            observation_config=observation_config,
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
                    f"action={np.asarray(action).tolist()} reward={reward:.4f} "
                    f"is_slipping={bool(info.get('is_slipping', False))} "
                    f"slip_recovered={bool(info.get('slip_recovered', False))} "
                    "force_delta_window=["
                    f"{float(info.get('left_force_delta_window_n', 0.0)):.4f}, "
                    f"{float(info.get('right_force_delta_window_n', 0.0)):.4f}] "
                    f"upward_action={float(info.get('upward_action_magnitude', 0.0)):.4f}"
                )
        print(f"termination_reason={info.get('reason') or 'unknown'}")

        if args.controller == "phys" and not getattr(controller, "_stopped", False):
            import rospy

            hold_command = controller.get_joint_angle_cmd()
            print("Holding final robot pose. Press Ctrl-C to stop.")
            try:
                while not rospy.is_shutdown():
                    controller.send_joint_angle_cmd(hold_command)
                    controller.step()
            except (KeyboardInterrupt, rospy.ROSInterruptException):
                pass
        elif args.controller == "sim" and not args.no_render:
            while True:
                env.controller.step()
    finally:
        if controller is not None and hasattr(controller, "stop"):
            controller.stop()


if __name__ == "__main__":
    main()
