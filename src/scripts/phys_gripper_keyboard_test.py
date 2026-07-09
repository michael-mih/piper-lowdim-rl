#!/usr/bin/env python3
from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rospy

from controllers.phys_controller import PhysController


def read_key() -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not ready:
        return None

    key = sys.stdin.read(1)
    if key == "\x1b":
        time.sleep(0.01)
        while select.select([sys.stdin], [], [], 0.0)[0]:
            key += sys.stdin.read(1)
    return key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual physical gripper keyboard test.")
    parser.add_argument("--increment", type=float, default=0.001)
    parser.add_argument("--min-gripper", type=float, default=0.0)
    parser.add_argument("--max-gripper", type=float, default=0.06)
    parser.add_argument("--print-period", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller = PhysController(pid_controllers=[])
    controller.set_initial_position(controller.get_joint_angles())
    command = controller.get_joint_angle_cmd()

    print("Up/down arrows adjust gripper. Press q to quit.")
    stdin_fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(stdin_fd)

    try:
        tty.setcbreak(stdin_fd)
        last_print = 0.0

        while not rospy.is_shutdown():
            controller.step()

            key = read_key()
            if key in ("q", "Q"):
                break
            if key == "\x1b[A":
                command[6] = min(args.max_gripper, command[6] + args.increment)
                controller.send_joint_angle_cmd(command)
            elif key == "\x1b[B":
                command[6] = max(args.min_gripper, command[6] - args.increment)
                controller.send_joint_angle_cmd(command)

            now = time.time()
            if now - last_print >= args.print_period:
                joints = controller.get_joint_angles()
                left_force = controller.get_force_left()
                right_force = controller.get_force_right()
                print(
                    f"joints={['{:.4f}'.format(value) for value in joints]} "
                    f"force=({left_force:.4f}, {right_force:.4f}) "
                    f"cmd_gripper={command[6]:.4f}"
                )
                last_print = now

    finally:
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_settings)
        controller.stop()


if __name__ == "__main__":
    main()
