#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

JOINT5_INDEX = 4
GRIPPER_INDEX = 6
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "piper_ros"
    / "src"
    / "piper_description"
    / "mujoco_model"
    / "tmp_scene.xml"
)


def read_key(stdin_fd: int) -> str | None:
    ready, _, _ = select.select([stdin_fd], [], [], 0.0)
    if not ready:
        return None

    key = os.read(stdin_fd, 1).decode(errors="ignore")
    if key == "\x1b":
        while select.select([stdin_fd], [], [], 0.02)[0]:
            key += os.read(stdin_fd, 1).decode(errors="ignore")
    return key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual gripper and joint 5 keyboard test."
    )
    parser.add_argument(
        "--controller",
        choices=("sim", "phys"),
        default="sim",
        help="Controller backend (default: sim).",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="MuJoCo XML path used by the simulation backend.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable the MuJoCo viewer.",
    )
    parser.add_argument("--can-channel", default="can0")
    parser.add_argument("--increment", type=float, default=0.0005)
    parser.add_argument("--min-gripper", type=float, default=0.01)
    parser.add_argument("--max-gripper", type=float, default=0.06)
    parser.add_argument("--min-joint5", type=float, default=-1.22)
    parser.add_argument("--max-joint5", type=float, default=1.22)
    parser.add_argument("--print-period", type=float, default=0.25)
    parser.add_argument("--gpr-model-path", default=None)
    parser.add_argument("--feedback-timeout", type=float, default=5.0)
    parser.add_argument("--max-feedback-age", type=float, default=0.1)
    parser.add_argument("--max-gpr-std", type=float, default=1.0)
    parser.add_argument("--hard-force-limit", type=float, default=5.0)
    parser.add_argument("--command-speed-percent", type=int, default=10)
    parser.add_argument(
        "--allow-hardware-motion",
        action="store_true",
        help="Required acknowledgement before physical commands are enabled.",
    )
    return parser.parse_args()


def make_controller(args: argparse.Namespace):
    if args.controller == "sim":
        from controllers.sim_controller import SimController

        return SimController(
            pid_controllers=[],
            model_path=args.model_path,
            render=not args.no_render,
        )

    from controllers.phys_controller import PhysController

    return PhysController(
        pid_controllers=[],
        channel=args.can_channel,
        gpr_model_path=args.gpr_model_path,
        feedback_timeout_s=args.feedback_timeout,
        max_feedback_age_s=args.max_feedback_age,
        max_gpr_std_n=args.max_gpr_std,
        hard_force_limit_n=args.hard_force_limit,
        command_speed_percent=args.command_speed_percent,
    )


def main() -> None:
    args = parse_args()
    if args.controller == "phys" and not args.allow_hardware_motion:
        raise RuntimeError(
            "Physical keyboard control requires --allow-hardware-motion after checking "
            "the workspace, E-stop, and feedback topics"
        )
    controller = make_controller(args)
    old_settings = None
    try:
        controller.set_initial_position(controller.get_joint_angles())
        command = controller.get_joint_angle_cmd()
        selected_joint = GRIPPER_INDEX

        if args.controller == "phys":
            import rospy

            should_stop = rospy.is_shutdown
        else:
            should_stop = lambda: False

        print(
            f"Controller: {args.controller}. "
            "Up/down arrows or W/S adjust the selected joint. "
            "Press Tab, Space, or M to switch between gripper and joint 5; q quits."
        )
        print("Selected: gripper")
        stdin_fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(stdin_fd)
        tty.setcbreak(stdin_fd)
        last_print = 0.0

        while not should_stop():
            controller.step()

            key = read_key(stdin_fd)

            if key in ("q", "Q"):
                break
            if key in ("\t", " ", "m", "M"):
                selected_joint = (
                    JOINT5_INDEX
                    if selected_joint == GRIPPER_INDEX
                    else GRIPPER_INDEX
                )
                selected_name = (
                    "joint 5" if selected_joint == JOINT5_INDEX else "gripper"
                )
                print(f"\nSelected: {selected_name}")
            elif key in ("\x1b[A", "\x1b[B", "w", "W", "s", "S"):
                direction = 1.0 if key in ("\x1b[A", "w", "W") else -1.0
                minimum, maximum = (
                    (args.min_joint5, args.max_joint5)
                    if selected_joint == JOINT5_INDEX
                    else (args.min_gripper, args.max_gripper)
                )
                command[selected_joint] = max(
                    minimum,
                    min(maximum, command[selected_joint] + direction * args.increment),
                )
                controller.send_joint_angle_cmd(command)

            now = time.time()
            if now - last_print >= args.print_period:
                joints = controller.get_joint_angles()
                left_force = controller.get_force_left()
                right_force = controller.get_force_right()
                print(
                    f"joint5={joints[JOINT5_INDEX]:.4f} "
                    f"gripper={joints[GRIPPER_INDEX]:.4f} "
                    f"cmd_joint5={command[JOINT5_INDEX]:.4f} "
                    f"cmd_gripper={command[GRIPPER_INDEX]:.4f} "
                    f"force=({left_force:.4f}, {right_force:.4f}) "
                )
                last_print = now

    finally:
        if old_settings is not None:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_settings)
        stop = getattr(controller, "stop", None)
        if stop is not None:
            stop()


if __name__ == "__main__":
    main()
