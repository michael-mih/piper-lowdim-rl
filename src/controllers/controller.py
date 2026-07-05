from __future__ import annotations

from abc import ABC, abstractmethod

class Controller(ABC):
    def __init__(self, pid_controllers):
        self.pid_controllers = pid_controllers
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"]  # 假设机械臂有6个关节 + gripper

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

    