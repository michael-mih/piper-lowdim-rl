from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

from controllers.controller import Controller
#from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config

import joblib
from collections import deque
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState
from piper_msgs.msg import PiperStatusMsg
from piper_msgs.srv import Enable, Gripper, GripperRequest
from std_srvs.srv import Trigger

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
    ROS_Freq = 200.0
    dt = 1.0 / ROS_Freq

    # Gripper and Arm feedback
    gripper_pos = 0.0
    gripper_eff = 0.0
    current_positions = [0.0] * 7  # Holds full arm state to prevent arm movement
    joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper']
    joint_bounds = [
        -2.618, 2.168,
        0.0, 3.14,
        -2.967, 0.0,
        -1.745, 1.745,
        -1.22, 1.22,
        -2.0944, 2.0944,
        0.0, 0.07,
    ]
    DEFAULT_GPR_MODEL_PATH = (
        Path(__file__).resolve().parents[2]
        / "feng"
        / "gpr_model_small_force_200Hz_5.pkl"
    )

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
        ROS_Freq = 200.0
        dt = 1.0 / ROS_Freq

        # Gripper and Arm feedback
        gripper_pos = 0.0
        gripper_eff = 0.0
        current_positions = [0.0] * 7  # Holds full arm state to prevent arm movement
        joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper']

        def _record_scalar(self, name, value):
            value = float(value)
            with self._feedback_lock:
                if not math.isfinite(value) or not 0.0 <= value <= 4095.0:
                    self._invalid_feedback[name] = f"{name} out of ADC range: {value!r}"
                    return
                setattr(self, name, value)
                self._feedback_times[name] = time.monotonic()
                self._invalid_feedback.pop(name, None)

        #ROS callbacks
        def FSR1_Cb(self, data):
            self._record_scalar("fsr1", data.data)

        def FSR2_Cb(self, data):
            self._record_scalar("fsr2", data.data)

        def LoadCell_Cb(self, data):
            value = float(data.data)
            with self._feedback_lock:
                if math.isfinite(value):
                    self.loadcell = value
                    self._feedback_times["loadcell"] = time.monotonic()

        def JointState_Cb(self, msg):
            try:
                indices = [msg.name.index(name) for name in self.joint_names]
                positions = [float(msg.position[index]) for index in indices]
            except (ValueError, IndexError, TypeError) as exc:
                with self._feedback_lock:
                    self._invalid_feedback["joints"] = f"Malformed joint feedback: {exc}"
                return
            with self._feedback_lock:
                if not all(math.isfinite(value) for value in positions):
                    self._invalid_feedback["joints"] = "Joint feedback contains a non-finite value"
                    return
                self.current_positions = positions
                self._feedback_times["joints"] = time.monotonic()
                self._invalid_feedback.pop("joints", None)

        def ArmStatus_Cb(self, msg):
            with self._feedback_lock:
                self.arm_status = msg
                self._feedback_times["arm_status"] = time.monotonic()

        def __init__(
            self,
            feedback_timeout_s,
            max_feedback_age_s,
            command_speed_percent,
            expected_can_channel,
        ):
            rospy.init_node("FSR_Piper_Combined", anonymous=True)
            self.fsr1 = 4095.0
            self.fsr2 = 4095.0
            self.loadcell = 0.0
            self.current_positions = [0.0] * 7
            self.arm_status = None
            self.max_feedback_age_s = float(max_feedback_age_s)
            self._feedback_lock = threading.RLock()
            self._feedback_times = {
                "fsr1": None,
                "fsr2": None,
                "loadcell": None,
                "joints": None,
                "arm_status": None,
            }
            self._invalid_feedback = {}
            self._stopped = False
            self.driver_auto_enable = None
            self.gripper_srv = None
            self.stop_srv = None
            self.reset_srv = None
            self.enable_srv = None

            # ROS Subscribers
            self.subscribers = [
                rospy.Subscriber('/fsr1', Float32, self.FSR1_Cb, queue_size=1),
                rospy.Subscriber('/fsr2', Float32, self.FSR2_Cb, queue_size=1),
                rospy.Subscriber('/loadcell', Float32, self.LoadCell_Cb, queue_size=1),
                rospy.Subscriber('/joint_states_single', JointState, self.JointState_Cb, queue_size=1),
                rospy.Subscriber('/arm_status', PiperStatusMsg, self.ArmStatus_Cb, queue_size=1),
            ]

            # Piper joint command publishers 
            self.joint_pub = rospy.Publisher('/joint_states', JointState, queue_size=1, tcp_nodelay=True)

            # Publish the prediction so other nodes can use it
            self.pred1_pub = rospy.Publisher('/gpr_prediction1', Float32, queue_size=10)
            self.pred2_pub = rospy.Publisher('/gpr_prediction2', Float32, queue_size=10)
            self.pred_avg_pub = rospy.Publisher('/gpr_prediction_avg', Float32, queue_size=10)

            try:
                for service_name in (
                    "/gripper_srv",
                    "/stop_srv",
                    "/reset_srv",
                    "/enable_srv",
                ):
                    rospy.wait_for_service(service_name, timeout=feedback_timeout_s)
                self.gripper_srv = rospy.ServiceProxy("/gripper_srv", Gripper)
                self.stop_srv = rospy.ServiceProxy("/stop_srv", Trigger)
                self.reset_srv = rospy.ServiceProxy("/reset_srv", Trigger)
                self.enable_srv = rospy.ServiceProxy("/enable_srv", Enable)

                self.wait_for_feedback(feedback_timeout_s)
                self.verify_can_channel(expected_can_channel)
            except Exception:
                self.stop()
                raise

            # Pre-allocate command payload. The bundled driver interprets
            # velocity[6] as the global arm speed percentage.
            self.cmd_msg = JointState()
            self.cmd_msg.name = self.joint_names
            self.cmd_msg.velocity = [0.0] * 6 + [float(command_speed_percent)]
            self.cmd_msg.effort = [0.0] * 7

            self.rate = rospy.Rate(self.ROS_Freq)
            self.T0 = rospy.get_time()

        def enable_gripper(self, target_angle=None):
            enable_req = GripperRequest()
            enable_req.gripper_angle = float(
                self.current_positions[6] if target_angle is None else target_angle
            )
            enable_req.gripper_effort = 0.5
            enable_req.gripper_code = 0x01
            enable_req.set_zero = 0
            try:
                response = self.gripper_srv(enable_req)
            except rospy.ServiceException as e:
                raise RuntimeError(f"Failed to enable gripper service: {e}") from e
            if not response.status:
                raise RuntimeError(f"Gripper enable was rejected with code {response.code}")

        def enable_arm(self):
            try:
                response = self.enable_srv(True)
            except rospy.ServiceException as e:
                raise RuntimeError(f"Failed to enable Piper arm: {e}") from e
            if not response.enable_response:
                raise RuntimeError("Piper arm enable service reported failure")

        def reset_arm(self):
            try:
                response = self.reset_srv()
            except rospy.ServiceException as e:
                raise RuntimeError(f"Failed to reset Piper arm stop state: {e}") from e
            if not response.success:
                raise RuntimeError(
                    f"Piper arm reset service reported failure: {response.message}"
                )

        def wait_for_feedback(self, timeout_s):
            required = ("fsr1", "fsr2", "joints", "arm_status")
            deadline = time.monotonic() + float(timeout_s)
            while not rospy.is_shutdown():
                with self._feedback_lock:
                    missing = [name for name in required if self._feedback_times[name] is None]
                if not missing:
                    self.assert_feedback_safe()
                    return
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Timed out waiting for physical feedback: " + ", ".join(missing)
                    )
                rospy.sleep(0.01)
            raise RuntimeError("ROS shut down while waiting for physical feedback")

        def verify_can_channel(self, expected_channel):
            can_params = {
                name: rospy.get_param(name)
                for name in rospy.get_param_names()
                if name.endswith("/can_port")
            }
            matching_params = [
                name for name, value in can_params.items() if value == expected_channel
            ]
            if not matching_params:
                raise RuntimeError(
                    f"No running Piper ROS driver reports can_port={expected_channel!r}; "
                    f"discovered CAN parameters: {can_params}"
                )
            auto_enable_params = {
                name.rsplit("/", 1)[0] + "/auto_enable": rospy.get_param(
                    name.rsplit("/", 1)[0] + "/auto_enable",
                    None,
                )
                for name in matching_params
            }
            invalid_values = {
                name: value
                for name, value in auto_enable_params.items()
                if not isinstance(value, bool)
            }
            if invalid_values:
                raise RuntimeError(
                    "Piper ROS driver auto_enable parameters must be booleans; "
                    f"discovered: {invalid_values}"
                )
            unique_values = set(auto_enable_params.values())
            if len(unique_values) != 1:
                raise RuntimeError(
                    "Multiple matching Piper ROS drivers disagree on auto_enable: "
                    f"{auto_enable_params}"
                )
            self.driver_auto_enable = unique_values.pop()
            rospy.loginfo(
                "Using Piper driver on %s with auto_enable=%s",
                expected_channel,
                self.driver_auto_enable,
            )

        def assert_feedback_safe(self):
            required = ("fsr1", "fsr2", "joints", "arm_status")
            now = time.monotonic()
            with self._feedback_lock:
                if self._invalid_feedback:
                    raise RuntimeError("; ".join(self._invalid_feedback.values()))
                stale = [
                    name
                    for name in required
                    if self._feedback_times[name] is None
                    or now - self._feedback_times[name] > self.max_feedback_age_s
                ]
                status = self.arm_status
            if stale:
                raise RuntimeError("Stale physical feedback: " + ", ".join(stale))
            if status is None:
                raise RuntimeError("Arm status feedback is unavailable")
            error_flags = [
                name
                for name in (
                    "joint_1_angle_limit", "joint_2_angle_limit", "joint_3_angle_limit",
                    "joint_4_angle_limit", "joint_5_angle_limit", "joint_6_angle_limit",
                    "communication_status_joint_1", "communication_status_joint_2",
                    "communication_status_joint_3", "communication_status_joint_4",
                    "communication_status_joint_5", "communication_status_joint_6",
                )
                if bool(getattr(status, name, False))
            ]
            if int(getattr(status, "err_code", 0)) != 0 or error_flags:
                raise RuntimeError(
                    f"Piper arm fault err_code={getattr(status, 'err_code', None)} "
                    f"flags={error_flags}"
                )

        def snapshot(self):
            self.assert_feedback_safe()
            with self._feedback_lock:
                return self.fsr1, self.fsr2, list(self.current_positions)

        def stop(self):
            if self._stopped:
                return
            self._stopped = True
            rospy.logwarn("Stopping Piper arm and disabling arm/gripper control")
            if self.stop_srv is not None:
                try:
                    self.stop_srv()
                except rospy.ServiceException as e:
                    rospy.logerr(f"Failed to stop Piper arm: {e}")

            disable_req = GripperRequest()
            disable_req.gripper_angle = 0.0
            disable_req.gripper_effort = 0.5
            disable_req.gripper_code = 0x00
            disable_req.set_zero = 0
            if self.gripper_srv is not None:
                try:
                    self.gripper_srv(disable_req)
                except rospy.ServiceException as e:
                    rospy.logerr(f"Failed to disable gripper: {e}")
            if self.enable_srv is not None:
                try:
                    response = self.enable_srv(False)
                    if not response.enable_response:
                        rospy.logerr("Piper arm disable service reported failure")
                except rospy.ServiceException as e:
                    rospy.logerr(f"Failed to disable Piper arm: {e}")







    def __init__(
        self,
        pid_controllers: Sequence[Any],
        ground_truth_pos: bool = False,
        channel: str = "can0",
        gpr_model_path: Optional[str] = None,
        feedback_timeout_s: float = 5.0,
        max_feedback_age_s: float = 0.1,
        max_gpr_std_n: float = 1.0,
        hard_force_limit_n: float = 5.0,
        command_speed_percent: int = 10,
    ) -> None:
        super().__init__(pid_controllers, ground_truth_pos)
        if not 1 <= int(command_speed_percent) <= 20:
            raise ValueError("command_speed_percent must be between 1 and 20")
        if feedback_timeout_s <= 0.0 or max_feedback_age_s <= 0.0:
            raise ValueError("Feedback timeout and maximum age must be positive")
        self.max_gpr_std_n = float(max_gpr_std_n)
        if self.max_gpr_std_n < 0.0:
            raise ValueError("max_gpr_std_n must be non-negative")
        self.hard_force_limit_n = float(hard_force_limit_n)
        if self.hard_force_limit_n <= 0.0:
            raise ValueError("hard_force_limit_n must be positive")
        model_path = Path(gpr_model_path) if gpr_model_path else self.DEFAULT_GPR_MODEL_PATH
        try:
            self.gpr = joblib.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load GPR force model {model_path}: {e}") from e

        self.N = int(getattr(self.gpr, "n_features_in_", 5))
        if self.N != 5:
            raise RuntimeError(f"Expected a five-sample GPR model, got {self.N} features")
        self.fsr_buffer1 = deque(maxlen=self.N)
        self.fsr_buffer2 = deque(maxlen=self.N)
        self.y_pred1_val = 0.0
        self.y_pred2_val = 0.0
        self._last_prediction_time = None
        self._stopped = False
        self._enabled = False
        self.max_arm_command_delta_rad = 0.02
        self.max_gripper_command_delta_m = 0.001
        self.max_arm_tracking_error_rad = 0.05
        self.max_gripper_tracking_error_m = 0.005

        self.ros = None
        try:
            self.ros = self.RosInterface(
                feedback_timeout_s=feedback_timeout_s,
                max_feedback_age_s=max_feedback_age_s,
                command_speed_percent=command_speed_percent,
                expected_can_channel=channel,
            )
            self.target_angles = self.get_joint_angles()
            for _ in range(self.N):
                self.step()
            rospy.on_shutdown(self.stop)
        except Exception:
            if self.ros is not None:
                self.ros.stop()
            raise

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._enabled = False
        if self.ros is not None:
            self.ros.stop()

    emergency_stop = stop


    def get_joint_angle_cmd(self) -> list[float]:
        return list(self.target_angles)

    def get_joint_angles(self) -> list[float]:
        self._assert_feedback_safe_or_stop()
        with self.ros._feedback_lock:
            positions = list(self.ros.current_positions)
        return positions

    def send_joint_angle_cmd(self, cmds: Sequence[float]) -> None:
        if self._stopped:
            raise RuntimeError("Physical controller is stopped")
        if not self._enabled:
            raise RuntimeError("Physical controller has not been safely initialized")
        if len(cmds) != 7:
            raise ValueError("Expected 7 command values (6 arm joints and 1 gripper).")

        self._assert_feedback_safe_or_stop()
        values = [float(value) for value in cmds]
        if not all(math.isfinite(value) for value in values):
            self.emergency_stop()
            raise RuntimeError("Refusing a non-finite physical joint command")
        for index, value in enumerate(values):
            lo, hi = self.joint_bounds[index * 2:index * 2 + 2]
            if not lo <= value <= hi:
                self.emergency_stop()
                raise RuntimeError(
                    f"Refusing out-of-bounds command for {self.joint_names[index]}: "
                    f"{value} not in [{lo}, {hi}]"
                )
        arm_delta = max(abs(values[index] - self.target_angles[index]) for index in range(6))
        gripper_delta = abs(values[6] - self.target_angles[6])
        if arm_delta > self.max_arm_command_delta_rad + 1e-9:
            self.emergency_stop()
            raise RuntimeError(f"Refusing arm command jump of {arm_delta:.6f} rad")
        if gripper_delta > self.max_gripper_command_delta_m + 1e-9:
            self.emergency_stop()
            raise RuntimeError(f"Refusing gripper command jump of {gripper_delta:.6f} m")
        with self.ros._feedback_lock:
            actual = list(self.ros.current_positions)
        arm_tracking_error = max(abs(values[index] - actual[index]) for index in range(6))
        gripper_tracking_error = abs(values[6] - actual[6])
        if arm_tracking_error > self.max_arm_tracking_error_rad:
            self.emergency_stop()
            raise RuntimeError(
                f"Refusing command with arm tracking error {arm_tracking_error:.6f} rad"
            )
        if gripper_tracking_error > self.max_gripper_tracking_error_m:
            self.emergency_stop()
            raise RuntimeError(
                f"Refusing command with gripper tracking error {gripper_tracking_error:.6f} m"
            )
        self.target_angles = values

        self.ros.cmd_msg.header.stamp = rospy.Time.now()
        self.ros.cmd_msg.position = values
        self.ros.cmd_msg.effort[6] = 1
        self.ros.joint_pub.publish(self.ros.cmd_msg)

    def set_initial_position(self, initial_pos: Sequence[float] = None) -> None:
        
        actual = self.get_joint_angles()
        requested = actual if initial_pos is None else [float(value) for value in initial_pos]
        if len(requested) != 7:
            raise ValueError("Expected seven initial joint values")
        arm_delta = max(abs(requested[index] - actual[index]) for index in range(6))
        gripper_delta = abs(requested[6] - actual[6])
        if arm_delta > self.max_arm_command_delta_rad or gripper_delta > 0.01:
            raise RuntimeError(
                "Physical startup pose must be reached manually or through a verified trajectory; "
                f"requested deltas were arm={arm_delta:.6f} rad, gripper={gripper_delta:.6f} m"
            )
        # Hold measured state rather than introducing a startup step.
        self.target_angles = list(actual)
        if not self._enabled:
            try:
                # Refresh the driver's private enable flag even when its startup
                # auto-enable option is true. This also handles a prior controller
                # instance having disabled the arm during shutdown.
                self.ros.reset_arm()
                self.ros.enable_arm()
                self.ros.enable_gripper(target_angle=actual[6])
                self._enabled = True
            except Exception:
                self.emergency_stop()
                raise
        self.send_joint_angle_cmd(actual)



    def step(self) -> None:
        if self._stopped:
            raise RuntimeError("Physical controller is stopped")
        self._assert_feedback_safe_or_stop()

        # GPR Prediction Setup
        fsr1, fsr2, _ = self.ros.snapshot()
        normalized_fsr1 = (4095.0 - fsr1) / 4095.0
        normalized_fsr2 = (4095.0 - fsr2) / 4095.0

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
            std1_val = std1_val[0]
            std2_val = std2_val[0]
            values = (y_pred1_val, y_pred2_val, std1_val, std2_val)
            if not all(math.isfinite(float(value)) for value in values):
                self.emergency_stop()
                raise RuntimeError("GPR force model returned a non-finite value")
            if max(float(std1_val), float(std2_val)) > self.max_gpr_std_n:
                self.emergency_stop()
                raise RuntimeError(
                    f"GPR uncertainty exceeded {self.max_gpr_std_n} N: "
                    f"left={std1_val}, right={std2_val}"
                )
            if min(float(y_pred1_val), float(y_pred2_val)) < -0.1:
                self.emergency_stop()
                raise RuntimeError(
                    f"GPR force estimate was implausibly negative: {y_pred1_val}, {y_pred2_val}"
                )
            self.y_pred1_val = float(y_pred1_val)
            self.y_pred2_val = float(y_pred2_val)
            self._last_prediction_time = time.monotonic()
            if max(self.y_pred1_val, self.y_pred2_val) >= self.hard_force_limit_n:
                self.emergency_stop()
                raise RuntimeError(
                    f"Physical force limit {self.hard_force_limit_n} N reached: "
                    f"left={self.y_pred1_val}, right={self.y_pred2_val}"
                )
            
            self.ros.pred1_pub.publish(Float32(self.y_pred1_val))
            self.ros.pred2_pub.publish(Float32(self.y_pred2_val))
        self.ros.rate.sleep()

    def get_force_left(self) -> float:
        self._assert_prediction_fresh()
        return self.y_pred1_val
    def get_force_right(self) -> float:
        self._assert_prediction_fresh()
        return self.y_pred2_val

    def _assert_prediction_fresh(self):
        if self._last_prediction_time is None:
            raise RuntimeError("Physical force prediction is not ready")
        if time.monotonic() - self._last_prediction_time > self.ros.max_feedback_age_s:
            self.emergency_stop()
            raise RuntimeError("Physical force prediction is stale")

    def _assert_feedback_safe_or_stop(self):
        try:
            self.ros.assert_feedback_safe()
        except Exception:
            self.emergency_stop()
            raise

    def _clamp(self, val, min, max):
        return max(min, min(val, max))
