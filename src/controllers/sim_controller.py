from controllers.controller import Controller

import math
import time
import glfw  # 用于检查窗口关闭事件

class SimController(Controller):
    def __init__(self, pid_controllers, model_path, ground_truth_pos: bool = False, render=True):
        from mujoco_py import load_model_from_path, MjSim

        super().__init__(pid_controllers=pid_controllers, ground_truth_pos=ground_truth_pos)
        model = load_model_from_path(model_path)
        self.sim = MjSim(model)
        self.viewer = None
        if render:
            from mujoco_py import MjViewer

            self.viewer = MjViewer(self.sim)
        self.joint_bounds = [-2.618, 2.168, 0, 3.14, -2.967, 0, -1.745, 1.745, -1.22, 1.22, -2.0944, 2.0944, 0, 0.035, -0.035, 0]
        self.target_angles = self.get_joint_angles()
        
        self._passive_joint_state = self._capture_passive_joint_state()
        self.drop_termination_height_m = 0.025
        self.drop_termination_offset_m = 1.0
        self.object_dropped = False

       



    def _capture_passive_joint_state(self):
        passive_joint_state = []
        controlled_joints = set(self.joint_names)
        for joint_id in range(self.sim.model.njnt):
            joint_name = self.sim.model.joint_id2name(joint_id)
            if joint_name in controlled_joints:
                continue

            qpos_addr = self.sim.model.jnt_qposadr[joint_id]
            qvel_addr = self.sim.model.jnt_dofadr[joint_id]
            qpos_len, qvel_len = self._joint_state_lengths(joint_id)
            passive_joint_state.append(
                (
                    qpos_addr,
                    qpos_len,
                    qvel_addr,
                    qvel_len,
                    self.sim.data.qpos[qpos_addr : qpos_addr + qpos_len].copy(),
                    self.sim.data.qvel[qvel_addr : qvel_addr + qvel_len].copy(),
                )
            )
        return passive_joint_state

    def _restore_passive_joint_state(self):
        for qpos_addr, qpos_len, qvel_addr, qvel_len, qpos, qvel in self._passive_joint_state:
            self.sim.data.qpos[qpos_addr : qpos_addr + qpos_len] = qpos
            self.sim.data.qvel[qvel_addr : qvel_addr + qvel_len] = qvel

    def _joint_state_lengths(self, joint_id):
        joint_type = self.sim.model.jnt_type[joint_id]
        if joint_type == 0:  # free joint
            return 7, 6
        if joint_type == 1:  # ball joint
            return 4, 3
        return 1, 1

    def _mark_dropped_passive_objects(self):
        if self.object_dropped:
            return

        for qpos_addr, qpos_len, qvel_addr, qvel_len, initial_qpos, _ in self._passive_joint_state:
            if qpos_len != 7:
                continue

            initial_z = initial_qpos[2]
            current_z = self.sim.data.qpos[qpos_addr + 2]
            if current_z >= initial_z - self.drop_termination_height_m:
                continue

            self.object_dropped = True
            self.sim.data.qpos[qpos_addr] = initial_qpos[0] + self.drop_termination_offset_m
            self.sim.data.qvel[qvel_addr : qvel_addr + qvel_len] = 0
            self.sim.forward()
            return

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
        self.object_dropped = False
        self._restore_passive_joint_state()
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
        self._mark_dropped_passive_objects()
        if self.viewer is not None:
            self.viewer.render()

    def get_sensor_value(self, sensor_name):
        sensor_id = self.sim.model.sensor_name2id(sensor_name)
        sensor_addr = self.sim.model.sensor_adr[sensor_id]
        return float(self.sim.data.sensordata[sensor_addr])

    def get_force_left(self):
        return self.get_sensor_value("left_gripper_force")

    def get_force_right(self):
        return self.get_sensor_value("right_gripper_force")
