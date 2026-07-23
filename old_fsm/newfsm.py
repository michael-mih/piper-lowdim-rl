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
    inital_config = None
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
    inital_config = controller.get_joint_angle_cmd()
    initial_force = controller.get_force_average()
    increment = 0.00001
    iteration = 0
    total_iterations = 4
    converge_sum = 0
    stop = False
    print("iteration 1")
    while iteration < total_iterations:
        fsmActor.step()
        if fsmActor.is_converged(10):
            #break
            pass
        if not stop:
            print("iteration " + str(iteration+1) + ", force: " + str(fsmActor.controller.get_force_average()))
        if fsmActor.is_slip():
            fsmActor.tighten(increment)
            val = (fsmActor.three_force_buffer_left[1] + fsmActor.three_force_buffer_right[1]) / 2
            print("slipped at " + str(val))
            converge_sum += val
            stop = True 
            controller.send_joint_angle_cmd(inital_config)
            i = 0
            while(i<2000):
                controller.step()
                i+=1
            iteration += 1
            
            stop = False
            #converge_sum += (fsmActor.three_force_buffer_left[1] + fsmActor.three_force_buffer_right[1]) / 2
            #fsmActor.min = None
            #while(fsmActor.controller.get_force_left() < initial_force or fsmActor.controller.get_force_right() < initial_force):
            #    fsmActor.tighten(increment)
            #    fsmActor.step()
            
            #iteration +=1
            #fsmActor.min = None
        elif not stop:
            fsmActor.loosen(increment)
        #print(controller.get_force_average())
        if True and glfw.window_should_close(controller.viewer.window):
            break 
        time.sleep(0.01)

    print("converged at avg " + str((converge_sum / total_iterations)))
    
if __name__ == "__main__":
    main()
