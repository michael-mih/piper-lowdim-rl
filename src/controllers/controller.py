from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

class Controller(ABC):
    def __init__(self, pid_controllers, ground_truth_pos):
        self.pid_controllers = pid_controllers
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"]  # 假设机械臂有6个关节 + gripper
        self.ground_truth_pos = ground_truth_pos
        self.force_error_integral = 0.0
        self.force_error_previous = 0.0
        self.force_derivative_filtered = 0.0
        self.force_pid_initialized = False
        self.force_error_integral_limit = 10.0
        self.max_force_pid_delta = 0.0005
    @abstractmethod
    def send_joint_angle_cmd(self, cmds: list[float]) -> None:
        pass
    


    #current command
    @abstractmethod
    def get_joint_angle_cmd(self) -> list[float]:
        pass
    
    #current actual position
    @abstractmethod
    def get_joint_angles(self) -> list[float]:
        pass
    
    @abstractmethod
    def set_initial_position(self, initial_pos: list[float]) -> None:
        pass

    @abstractmethod
    def step(self) -> None:
        pass

    @abstractmethod
    def get_force_left(self) -> float:
        pass
    
    @abstractmethod
    def get_force_right(self) -> float:
        pass

    def get_force_average(self) -> float:
        return (self.get_force_left() + self.get_force_right()) / 2

    def pid(self, F_desired: float, dt: float) -> float:
        """Return the force-PID gripper position adjustment for one update."""
        if dt <= 0.0:
            raise ValueError("dt must be greater than zero")

        kp = 0.0000255
        ki = 0.000004
        kd = 0.000003
        alpha = 0.25

        force_feedback = self.get_force_average()
        force_error = F_desired - force_feedback

        if not np.isfinite(force_error):
            raise RuntimeError("Force PID received a non-finite force error")

        self.force_error_integral = float(
            np.clip(
                self.force_error_integral + force_error * dt,
                -self.force_error_integral_limit,
                self.force_error_integral_limit,
            )
        )
        derivative_raw = (
            (force_error - self.force_error_previous) / dt
            if self.force_pid_initialized
            else 0.0
        )
        self.force_derivative_filtered = (
            alpha * derivative_raw
            + (1.0 - alpha) * self.force_derivative_filtered
        )

        delta_angle = (
            kp * force_error
            + ki * self.force_error_integral
            + kd * self.force_derivative_filtered
        )
        self.force_error_previous = force_error
        self.force_pid_initialized = True
        return float(
            np.clip(
                delta_angle,
                -self.max_force_pid_delta,
                self.max_force_pid_delta,
            )
        )

    def reset_force_pid(self) -> None:
        self.force_error_integral = 0.0
        self.force_error_previous = 0.0
        self.force_derivative_filtered = 0.0
        self.force_pid_initialized = False
