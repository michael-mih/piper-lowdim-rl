from __future__ import annotations

import math
import os
import threading
import time
from typing import Any, Callable, Optional, Sequence, Tuple

from controllers.controller import Controller
from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config

import joblib
from collections import deque
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState
from piper_msgs.srv import Gripper, GripperRequest

import rospy
import numpy as np

class PhysController(Controller):

    #SUB TO: fsr1 fsr2 joint_states_single #
    #0-1495 analog into grp_model 
    #PUB" 


    is_physical = True

    # FSR + Loadcell
    fsr1 = 4095.0
    fsr2 = 4095.0
    loadcell = 0.0

    # ROS frequency
    ROS_Freq = 100.0
    dt = 1.0 / ROS_Freq

    # Gripper and Arm feedback
    gripper_pos = 0.0
    gripper_eff = 0.0
    current_positions = [0.0] * 7  # Holds full arm state to prevent arm movement
    joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper']

    CONTROL_MODE = 'pid'

    def gripper_force_ref(self, t):
    # Sine wave setup
        amplitude = 3
        frequency = 0.25
        bias = 7
        F_desired = amplitude * math.sin(frequency * t - math.pi / 2) + bias
        return F_desired


    def gripper_force_controller(self, mode, F_error, F_error_Int, d_error_filtered):
        """
        Consolidated force control function supporting standard 'pid' and 'saturated' profiles.
        """
        if mode == 'pid':
            Kp = 0.0000255
            Ki = 0.000004
            Kd = 0.000003
            delta_angle = Kp * F_error + Ki * (F_error_Int + F_error * self.dt) + Kd * d_error_filtered
            return delta_angle

        elif mode == 'saturated':
            # Saturated PI-D Control Algorithm from image law
            K_pi = 0.000085
            K_d  = 0.000005
            T = 10          # Integral time constant parameter
            acc = 50.0         # Acceleration dynamic limit bound
            omega_max = 20.0   # Velocity limit tracker bound

            K_pi = 0.000180
            K_d  = 0.000002 
            T = 10          
            acc = 100.0         
            omega_max = 50.0  
            
            # Pre-derived coefficient ratio: c / 2k = K_d / K_pi
            c_over_2k = K_d / K_pi
            
            # Compute dynamic saturation boundary limit
            L = c_over_2k * min(math.sqrt(4.0 * acc * abs(F_error)), omega_max)       
            
            # Clamp core PI components
            pi_block = F_error + (1.0 / T) * F_error_Int
            saturated_pi = np.clip(pi_block, -L, L)
            
            # Compute output matrix adjustment
            delta_angle = K_pi * saturated_pi + K_d * d_error_filtered
            return delta_angle

        else:
            # Fallback security bound
            return None

    class RosInterface:
        fsr1 = 4095.0
        fsr2 = 4095.0
        loadcell = 0.0

        # ROS frequency
        ROS_Freq = 100.0
        dt = 1.0 / ROS_Freq

        # Gripper and Arm feedback
        gripper_pos = 0.0
        gripper_eff = 0.0
        current_positions = [0.0] * 7  # Holds full arm state to prevent arm movement
        joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper']

        #ROS callbacks
        def FSR1_Cb(self, data):
            self.fsr1 = data.data

        def FSR2_Cb(self, data):
            self.fsr2 = data.data

        def LoadCell_Cb(self, data):
            self.loadcell = data.data

        def JointState_Cb(self, msg):
            if len(msg.position) >= 7:
                self.current_positions[0:7] = msg.position[0:7]
                

        

        def __init__(self):
            
            rospy.init_node("FSR_Piper_Combined")
            self.fsr1 = 4095.0
            self.fsr2 = 4095.0
            self.loadcell = 0.0
            self.current_positions = [0.0] * 7

            # ROS Subscribers
            rospy.Subscriber('/fsr1', Float32, self.FSR1_Cb)
            rospy.Subscriber('/fsr2', Float32, self.FSR2_Cb)
            rospy.Subscriber('/loadcell', Float32, self.LoadCell_Cb)
            rospy.Subscriber('/joint_states_single', JointState, self.JointState_Cb)

            # Piper joint command publishers 
            self.joint_pub = rospy.Publisher('/joint_states', JointState, queue_size=1, tcp_nodelay=True)

            # Publish the prediction so other nodes can use it
            self.pred1_pub = rospy.Publisher('/gpr_prediction1', Float32, queue_size=10)
            self.pred2_pub = rospy.Publisher('/gpr_prediction2', Float32, queue_size=10)
            self.pred_avg_pub = rospy.Publisher('/gpr_prediction_avg', Float32, queue_size=10)

            # Gripper Service Client (For Enable/Disable Only)
            rospy.wait_for_service("/gripper_srv")
            self.gripper_srv = rospy.ServiceProxy("/gripper_srv", Gripper)


            rospy.loginfo("Waiting 1.5 seconds for sensor data...")
            time.sleep(1.5)

            # Enable Gripper
            enable_req = GripperRequest()
            enable_req.gripper_angle = 0.05
            enable_req.gripper_effort = 0.0
            enable_req.gripper_code = 0x01
            enable_req.set_zero = 0

            rospy.loginfo("Enabling gripper control...")
            try:
                self.gripper_srv(enable_req)
            except rospy.ServiceException as e:
                rospy.logerr(f"Failed to enable gripper service: {e}")
                exit()


            rospy.loginfo("Starting combined 200 Hz topic-streaming loop...")

            # Pre-allocate message data payload layout to optimize 200Hz execution speed
            self.cmd_msg = JointState()
            self.cmd_msg.name = self.joint_names
            self.cmd_msg.velocity = [0.0] * 7
            self.cmd_msg.effort = [0.0] * 7

            # Main Loop (200 Hz Topic Streaming)
            self.rate = rospy.Rate(self.ROS_Freq)
            self.T0 = rospy.get_time()   

        def stop(self):
            disable_req = GripperRequest()
            disable_req.gripper_angle = 0.0
            disable_req.gripper_effort = 0.0
            disable_req.gripper_code = 0x00
            disable_req.set_zero = 0

            rospy.loginfo("Disabling gripper control...")
            try:
                self.gripper_srv(disable_req)
            except rospy.ServiceException as e:
                rospy.logerr(f"Failed to communicate gripper breakdown request: {e}")







    def __init__(
        self,
        pid_controllers: Sequence[Any],
        ground_truth_pos: bool = False,
        channel: str = "can0",
        
    ) -> None:
        super().__init__(pid_controllers, ground_truth_pos)



        self._connected = False
        self._enabled = False

        self.ros = self.RosInterface()

        try:
            self.gpr = joblib.load("gpr_model_small_force_200Hz_5.pkl")
            rospy.loginfo("GPR Model loaded successfully.")
        except Exception as e:
            rospy.logerr(f"Failed to load model: {e}")
            self.stop()

        self.N = 5
        self.fsr_buffer1 = deque(maxlen=self.N)
        self.fsr_buffer2 = deque(maxlen=self.N)

        self.F_error_Int = 0.0
        self.F_error_prev = 0.0      
        self.d_error_filtered = 0.0  
        self.alpha = 0.25
        self.y_pred1_val = 0.0
        self.y_pred2_val = 0.0
        self.target_angles = self.get_joint_angles()



      
    def stop(self) -> None:
        self.ros.stop()


    def get_joint_angle_cmd(self) -> list[float]:
        return list(self.target_angles)

    def get_joint_angles(self) -> list[float]:
        return list(self.ros.current_positions)

    def send_joint_angle_cmd(self, cmds: Sequence[float]) -> None:
        if len(cmds) != 7:
            raise ValueError("Expected 7 command values (6 arm joints and 1 gripper).")

        #clamped_cmds = []
        #for i in range(0, len(cmds)): #clamping
        #    clamped_cmds.append(max(self.joint_bounds[i*2], min(cmds[i], self.joint_bounds[i*2+1])))
        
        cmds[4] = self._clamp(cmds[4], 0, 0) #TODO
        cmds[6] = self._clamp(cmds[6], 0.0, 0.07)
        self.target_angles = list(cmds)

        self.ros.cmd_msg.header.stamp = rospy.Time.now()
        self.ros.cmd_msg.position = cmds
        self.ros.cmd_msg.effort[6] = 1
        self.ros.cmd_msg.velocity[6] = 0.0
        self.ros.joint_pub.publish(self.ros.cmd_msg)

    def set_initial_position(self, initial_pos: Sequence[float] = None) -> None:
        
        if initial_pos is not None:  
            self.target_angles = initial_pos
        else:
            self.target_angles = self.get_joint_angles()

        self.send_joint_angle_cmd(self.target_angles)



    def step(self) -> None:
        t = rospy.get_time() - self.ros.T0

        # GPR Prediction Setup
        normalized_fsr1 = (4095.0 - self.ros.fsr1) / 4095.0
        normalized_fsr2 = (4095.0 - self.ros.fsr2) / 4095.0

        self.fsr_buffer1.append(normalized_fsr1)
        self.fsr_buffer2.append(normalized_fsr2)

        y_pred1_val = self.y_pred1_val
        y_pred2_val = self.y_pred2_val
        std1_val = 0.0
        std2_val = 0.0

        if len(self.fsr_buffer1) == self.N:
            X1 = np.array(list(self.fsr_buffer1))[::-1].reshape(1, -1)
            X2 = np.array(list(self.fsr_buffer2))[::-1].reshape(1, -1)

            y_pred1_val, std1_val = self.gpr.predict(X1, return_std=True)
            y_pred2_val, std2_val = self.gpr.predict(X2, return_std=True)
            
            # Extract the scalar values out of the returned arrays
            y_pred1_val = y_pred1_val[0]
            y_pred2_val = y_pred2_val[0]
            self.y_pred1_val = y_pred1_val
            self.y_pred2_val = y_pred2_val
            std1_val = std1_val[0]
            std2_val = std2_val[0]
            
            self.ros.pred1_pub.publish(Float32(y_pred1_val))
            self.ros.pred2_pub.publish(Float32(y_pred2_val))



        #if t > 500.0:
        #    self.stop()

        
        self.ros.rate.sleep()

    def get_force_left(self) -> float:
        return self.y_pred1_val
    def get_force_right(self) -> float:
        return self.y_pred2_val

    def _clamp(self, val, min, max):
        return max(min, min(val, max))