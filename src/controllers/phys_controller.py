from controllers.controller import Controller
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW
import time
class PhysController(Controller):
    def __init__(self, pid_controllers, ground_truth_pos: bool = False, channel: str = "can0", speed_percent: int = 50):
        super().__init__(pid_controllers, ground_truth_pos)
        #TODO ???
        self.joint_bounds = [-2.618, 2.168, 0, 3.14, -2.967, 0, -1.745, 1.745, -1.22, 1.22, -2.0944, 2.0944, 0, 0.07]

        
        self.cfg = create_agx_arm_config(robot=ArmModel.PIPER, firmeware_version=PiperFW.DEFAULT, channel=channel)
        self.robot = AgxArmFactory.create_arm(self.cfg)
        self.end_effector = self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        
        self.robot.connect()
        if not self.robot.is_ok():
            raise Exception("Connection not ok")
        
        self.robot.set_speed_percent(speed_percent)
        self.initial_position = self.get_joint_angles()
        self.send_joint_angle_cmd(self.initial_position)

        while not self.robot.enable():
            time.sleep(0.01)
    
    
    def stop(self):
        self.robot.disconnect()

    def get_joint_angle_cmd(self):
        return self.target_angles

    def get_joint_angles(self):
        joint_angles = self.robot.get_joint_angles()
        if joint_angles is not None:
            current_angles = list(joint_angles.msg)
        else:
            current_angles = list(getattr(self, "target_angles", [0.0] * 7)[:6])

        gripper_status = self.end_effector.get_gripper_status()
        if gripper_status is not None:
            gripper_width = gripper_status.msg.value
        else:
            gripper_width = getattr(self, "target_angles", [0.0] * 7)[6]
        current_angles.append(gripper_width)
        current_angles.append(-gripper_width)
        return current_angles

    def send_joint_angle_cmd(self, cmds):
        if len(cmds) != len(self.joint_names):
            raise ValueError(
                f"Expected {len(self.joint_names)} command values "
                f"(6 arm joints + gripper width), got {len(cmds)}."
            )
        clamped_cmds = []
        for i in range(0, len(cmds)): #clamping
            clamped_cmds.append(max(self.joint_bounds[i*2], min(cmds[i], self.joint_bounds[i*2+1])))
        self.target_angles = clamped_cmds

    def set_initial_position(self, initial_pos):
        self.send_joint_angle_cmd(initial_pos)
        #TODO: wait for box?

    def step(self):
        for i in range(1, 7):
            self.robot.move_mit(
                joint_index=i,
                p_des=self.target_angles[i-1],
                v_des=0.0,
                kp=10.0,
                kd=0.8,
                t_ff=0.0,
            )
        self.end_effector.move_gripper_m(value=self.target_angles[6], force=1.0) #TODO?
        time.sleep(0.05)


    #TODO
    def get_sensor_value(self, sensor_name):
        pass

    def get_force_left(self):
        return 0.0
    def get_force_right(self):
        return 0.0
