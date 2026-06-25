from controllers.sim_controller import SimController
from scripts.build_sim import combined_xml
from fsm.custom_min_grasp_fsm import FSMActor
import os 
import argparse

import glfw  # 用于检查窗口关闭事件
import time
import asyncio

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
    
    fsmActor = FSMActor(controller=controller)
    count = 0
    target_angles = [0, 1.5, -0.3, 0, -0.7, 0, 0.03, -0.03]
    controller.set_initial_position(target_angles)
    gripper_delta = 0.0001
    arm_delta = 0.001
    while target_angles[4]>-1.2:
        controller.send_joint_angle_cmd(target_angles)
        controller.step()
        if min(controller.get_force_left(), controller.get_force_right()) > 0.8:
            gripper_delta = 0
            #example joint 5 movement
            #target_angles[4] = -1.2
        if gripper_delta == 0:
            if target_angles[4] > -1.2:
                target_angles[4]-=arm_delta
        #example gripper movement
        target_angles[6] -= gripper_delta
        target_angles[7] += gripper_delta
        if True and glfw.window_should_close(controller.viewer.window):
            break 
        
        time.sleep(0.01)

    
    increment = 0.00001
    while True:
        fsmActor.step()
        if fsmActor.is_converged(10):
            break
        if fsmActor.is_slip():
            fsmActor.tighten(increment)
        else:
            fsmActor.loosen(increment)
        #print(controller.get_force_average())
        if True and glfw.window_should_close(controller.viewer.window):
            break 
        time.sleep(0.01)

    print("converged at " + str(fsmActor.min))
    
if __name__ == "__main__":
    main()
