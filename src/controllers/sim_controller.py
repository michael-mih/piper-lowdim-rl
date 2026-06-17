from controllers.controller import Controller

import math
import time
import glfw  # 用于检查窗口关闭事件

class SimController(Controller):
    def __init__(self, pid_controllers, model_path):
        from mujoco_py import load_model_from_path, MjSim, MjViewer

        super().__init__(pid_controllers=pid_controllers)
        model = load_model_from_path(model_path)
        self.sim = MjSim(model)
        self.viewer = MjViewer(self.sim)
        self.joint_bounds = [-2.618, 2.168, 0, 3.14, -2.967, 0, -1.745, 1.745, -1.22, 1.22, -2.0944, 2.0944, 0, 0.035, -0.035, 0]



    def send_joint_angle_cmd(self, cmds):
        clamped_cmds = []
        for i in range(0, len(cmds)): #clamping
            clamped_cmds.append(max(self.joint_bounds[i*2], min(cmds[i], self.joint_bounds[i*2+1])))
        self.target_angles = clamped_cmds
    
    def get_joint_angle_cmd(self):
        return self.target_angles
    
    def get_joint_angles(self):
        current_angles = []
        for joint_name in self.joint_names:
            joint_id = self.sim.model.joint_name2id(joint_name)
            qpos_addr = self.sim.model.jnt_qposadr[joint_id]
            current_angles.append(self.sim.data.qpos[qpos_addr])  # 获取每个关节的当前角度
        return current_angles

    def set_initial_position(self, initial_pos):
        self.send_joint_angle_cmd(initial_pos)
        for i in range(len(self.joint_names)):
            joint_id = self.sim.model.joint_name2id(self.joint_names[i])
            qpos_addr = self.sim.model.jnt_qposadr[joint_id]
            self.sim.data.qpos[qpos_addr] = self.target_angles[i]
            self.sim.data.ctrl[i] = self.target_angles[i]

        self.sim.data.qvel[:] = 0
        self.sim.forward()

    def step(self):
        for i in range(len(self.joint_names)):
            self.sim.data.ctrl[i] = self.target_angles[i]
        self.sim.step()
        self.viewer.render()

    def get_sensor_value(self, sensor_name):
        sensor_id = self.sim.model.sensor_name2id(sensor_name)
        sensor_addr = self.sim.model.sensor_adr[sensor_id]
        return float(self.sim.data.sensordata[sensor_addr])

    def get_force_left(self):
        return self.get_sensor_value("left_gripper_force")

    def get_force_right(self):
        return self.get_sensor_value("right_gripper_force")
