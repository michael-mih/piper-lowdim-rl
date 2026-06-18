from controllers.sim_controller import SimController
from scripts.build_sim import combined_xml

import os 
import argparse

import glfw  # 用于检查窗口关闭事件
import time

# 定义PID控制器类
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_error = 0
        self.prev_error = 0
        self.dt = 0.01  # 控制周期，需要与模拟步长相匹配

    def calculate(self, target, current):
        error = target - current
        self.integral_error += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = (self.Kp * error) + (self.Ki * self.integral_error) + (self.Kd * derivative)
        self.prev_error = error
        return output


def main():
    parser = argparse.ArgumentParser(description="policy training args")
    parser.add_argument("-s", "--sim", action="store_true", help="sim piper in mujoco")
    
    current_path = os.path.dirname(os.path.realpath(__file__))
    
    
    args = parser.parse_args()

    
    if True: #if args.sim
        #current_path = os.path.dirname(os.path.realpath(__file__))
        #model_path = os.path.join(current_path, '..', '..', 'piper_ros', 'src', 'piper_description', 'mujoco_model', 'piper_description.xml')
        model_path = str(combined_xml)
        controller = SimController(pid_controllers=[PIDController(0.01, 0, 0) for _ in range(8)], model_path=model_path)
    else:
        pass
    
    count = 0
    target_angles = [0, 1.5, -0.3, 0, -1.0, 0, 0.03, -0.03]
    controller.set_initial_position(target_angles)
    gripper_delta = 0.0001
    while 1:
        controller.send_joint_angle_cmd(target_angles)
        #example gripper movement
        target_angles[6] -= gripper_delta
        target_angles[7] += gripper_delta

        controller.step()
        if True and glfw.window_should_close(controller.viewer.window):
            break 
        if controller.get_force_left() > 1.2:
            gripper_delta = 0
            #example joint 5 movement
            target_angles[4] = -1.2
        time.sleep(0.01)
        
        
