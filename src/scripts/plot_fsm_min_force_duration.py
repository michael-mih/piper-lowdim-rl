from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import List, Tuple

from controllers.sim_controller import SimController
from fsm.custom_min_grasp_fsm import FSMActor
from scripts.build_sim import combined_xml


Sample = Tuple[float, float]


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
    parser = argparse.ArgumentParser(description="Run the FSM grasp test and plot current force over elapsed time.")
    parser.add_argument("--model-path", default=str(combined_xml), help="MuJoCo XML path.")
    parser.add_argument("--plot-path", default="fsm_current_force_duration.png")
    parser.add_argument("--csv-path", default="fsm_current_force_duration.csv")
    parser.add_argument("--render", action="store_true", help="Render the FSM simulation.")
    parser.add_argument("--increment", type=float, default=0.00001)
    parser.add_argument("--convergence-interval", type=float, default=10.0)
    parser.add_argument("--sleep-s", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--prep-max-steps", type=int, default=5000)
    return parser.parse_args()


def make_controller(model_path: str, render: bool) -> SimController:
    return SimController(
        pid_controllers=[PIDController(0.01, 0.0, 0.0) for _ in range(8)],
        model_path=model_path,
        render=render,
    )


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
    controller = make_controller(args.model_path, render=args.render)
    fsm_actor = FSMActor(controller=controller)
    samples: List[Sample] = []
    start_time = time.perf_counter()

    target_angles = [0, 1.5, -0.3, 0, -0.7, 0, 0.03, -0.03]
    controller.set_initial_position(target_angles)
    gripper_delta = 0.0001
    arm_delta = 0.001

    prep_steps = 0
    while target_angles[4] > -1.2 and prep_steps < args.prep_max_steps:
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
        maybe_sleep(args.sleep_s)

    if target_angles[4] > -1.2:
        print(f"fsm_prep_stopped_after={prep_steps} target_joint5={target_angles[4]:.3f}")

    for _ in range(args.max_steps):
        fsm_actor.step()
        samples.append((time.perf_counter() - start_time, float(controller.get_force_average())))

        if fsm_actor.is_converged(args.convergence_interval):
            break
        if fsm_actor.is_slip():
            fsm_actor.tighten(args.increment)
        else:
            fsm_actor.loosen(args.increment)

        if viewer_closed(controller):
            break
        maybe_sleep(args.sleep_s)

    print(f"fsm_converged_min_force={fsm_actor.min}")
    return samples


def write_csv(path: Path, samples: List[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["elapsed_s", "current_force_n"])
        for elapsed_s, force_n in samples:
            writer.writerow([f"{elapsed_s:.6f}", f"{force_n:.6f}"])


def plot_samples(path: Path, samples: List[Sample]) -> None:
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

    if samples:
        x, y = zip(*samples)
        ax.plot(x, y, label="FSM current average force")
        ax.legend()
    else:
        print("warning: FSM produced no force samples, so no force curve was plotted.")
        ax.text(0.5, 0.5, "No force samples recorded", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Elapsed wall-clock time (s)")
    ax.set_ylabel("Current average gripper force (N)")
    ax.set_title("FSM current force vs program duration")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    samples = run_fsm(args)

    csv_path = Path(args.csv_path)
    plot_path = Path(args.plot_path)
    write_csv(csv_path, samples)
    plot_samples(plot_path, samples)
    print(f"wrote_csv={csv_path}")
    print(f"wrote_plot={plot_path}")


if __name__ == "__main__":
    main()
