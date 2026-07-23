from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
from controllers.controller import Controller

import numpy as np
import math



@dataclass
class ObservationConfig:
    min_force_threshold_n: float = 0.05
    slip_drop_threshold_n: float = 0.25
    slip_recovery_drop_tolerance_n: float = 0.10
    slip_recovery_force_ratio: float = 0.75
    slip_recovery_confirmation_steps: int = 2
    slip_rearm_confirmation_steps: int = 3
    calibration_max_steps: int = 10000
    calibration_force_mode: str = "both"
    normalize: bool = True
    max_force_n: float = 5.0
    max_force_delta_n: float = 1.0
    max_stiffness_n_per_m: float = 1000.0
    max_gripper_gap_m: float = 0.07
    max_gripper_gap_delta_m: float = 0.005
    min_gripper_gap_m: float = 0.01
    max_force_error_integral_ns: float = 10.0
    max_force_derivative_n_per_s: float = 200.0
    gripper_gap_epsilon_m: float = 1e-4
    sensor_offset_range: float = 0.02
    sensor_noise_std: float = 0.01
    motor_offset_range: float = 0.01

## GPR force estimate -> model returns desired delta
## GPR force estimate -> model returns desired delta fed into PID 
## GPR Force estimate -> model learns desired delta as well as PID coefficients?


@dataclass
class RewardConfig:
    desired_force_min_n: float = 0.1
    desired_force_max_n: float = 3.5
    terminate_force_n: float = 5.0
    force_penalty: float = -0.0004
    slip_lift_penalty: float = -0.001
    slip_recovery_bonus: float = 0.002
    slip_recovery_bonus_min_lift_progress: float = 0.05
    lift_reward_height_m: float = 0.005
    lift_reward: float = 0.0008
    high_lift_reward_height_m: float = 0.015
    high_lift_reward: float = 0.0016
    success_lift_height_m: float = 0.040
    success_reward: float = 1.0
    failure_penalty: float = -1.0
    step_penalty: Optional[float] = None
    out_of_bounds_radius_m: float = 0.25
    success_force_max_n: float = 2.0
    peak_force_penalty_coef: float = 0.5
    average_force_penalty_coef: float = 0.2
    force_drop_tolerance_n: float = 0.1
    force_drop_penalty_coef: float = 0.01
    lift_progress_reward_coef: float = 1.0
    contact_force_scale_n: float = 0.3
    force_change_reward_coef: float = 0.02
    max_rewarded_force_change_n: float = 0.25

    ideal_force_mass_ratio = 0.15 / 0.05
    ratio_tolerance = 0.1

    #low_lift_angle: float = -0.80
    #high_lift_angle: float = -0.86
    #success_lift_angle: float = -0.98

    low_lift_angle: float = -0.80
    high_lift_angle: float = -0.86
    success_lift_angle: float = -1.10



@dataclass
class GraspEnvConfig:
    initial_joint_angles: Tuple[float, ...] = (0, 1.5, -0.3, 0, -0.7, 0, 0.04) #start w open gripper and close via calibration ONLY
    max_steps: int = 250
    settle_steps: int = 5
    control_steps_per_action: Optional[int] = None
    control_period_s: float = 0.02
    gripper_delta_m: float = 0.0001
    joint5_lift_delta_rad: float = -0.01
    joint5_lower_delta_rad: float = 0.01
    calibration_gripper_delta: float = 0.00001
    max_gripper_control_delta_m: float = 0.0005
    physical_start_pose_tolerance_rad: float = 0.05
    physical_start_gripper_tolerance_m: float = 0.015

    desired_force_min_n: float = 0.001
    desired_force_max_n: float = 4.0
    relative_force_drop_epsilon_n: float = 0.1

    
    joint5_index: int = 4
    gripper_index: int = 6
    default_body_name: str = "grasp_box"


ObjectHeightFn = Callable[[Any], Optional[float]]
ObjectPositionFn = Callable[[Any], Optional[Sequence[float]]]


class GraspPPOEnv:
    """Small controller-backed grasping environment for discrete PPO.

    The environment deliberately avoids a Gym dependency. It exposes a familiar
    reset/step interface and keeps robot-specific assumptions configurable.
    """

    num_actions = 2  # Continuous action vector: [joint5 delta, desired force].
    policy_schema_version = 5
    observation_names = (
        "left_force_n",
        "right_force_n",
        "left_force_delta_n",
        "right_force_delta_n",
        "gripper_gap_m",
        "gripper_gap_delta_m",
        "joint5_position",
        "force_error_integral_ns",
        "force_derivative_filtered_n_per_s",
        "is_slipping",
    )

    def __init__(
        self,
        controller: Controller,
        env_config: Optional[GraspEnvConfig] = None,
        observation_config: Optional[ObservationConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        object_height_fn: Optional[ObjectHeightFn] = None,
        object_position_fn: Optional[ObjectPositionFn] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.controller = controller
        self.env_config = env_config or GraspEnvConfig()
        self.observation_config = observation_config or ObservationConfig()
        self.reward_config = reward_config or RewardConfig()
        self.object_height_fn = object_height_fn
        self.object_position_fn = object_position_fn
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.done = False
        self.stiffness_n_per_m = 0.0
        self.last_info: Dict[str, Any] = {}
        self._initial_object_height: Optional[float] = None
        self._initial_object_xy: Optional[np.ndarray] = None
        self._sensor_offset = 0.0
        self._motor_offset = 0.0
        self._previous_left_force = 0.0
        self._previous_right_force = 0.0
        self._left_force_delta_buffer = deque((0.0, 0.0, 0.0), maxlen=3)
        self._right_force_delta_buffer = deque((0.0, 0.0, 0.0), maxlen=3)
        self._is_slipping_state = False
        self._slip_recovered_this_step = False
        self._slip_recovery_steps = 0
        self._slip_detection_armed = True
        self._slip_rearm_steps = 0
        self._initial_joint5_position: Optional[float] = None
        self._episode_max_lift_progress = 0.0
        self._last_recovery_bonus_progress = 0.0
        self._previous_gripper_gap = 0.0
        self._previous_desired_force_n = self.env_config.desired_force_min_n
        self._max_force_drop_n = 0.0
        self._max_relative_force_drop = 0.0
        self._sustained_force_drop_n = 0.0
        self._joint5_lift_progress_rad = 0.0
        self._lifting_this_step = False
        self._upward_action_magnitude = 0.0
        self.episode_force_sum = 0.0
        self.episode_force_samples = 0
        self.episode_peak_force = 0.0


    @property
    def observation_dim(self) -> int:
        return len(self.observation_names)

    def policy_metadata(self) -> Dict[str, Any]:
        return self.policy_metadata_for_configs(
            self.env_config,
            self.observation_config,
        )

    @classmethod
    def policy_metadata_for_configs(
        cls,
        env_config: GraspEnvConfig,
        observation_config: ObservationConfig,
    ) -> Dict[str, Any]:
        return {
            "schema_version": cls.policy_schema_version,
            "observation_names": list(cls.observation_names),
            "normalize": bool(observation_config.normalize),
            "calibration": {
                "min_force_threshold_n": observation_config.min_force_threshold_n,
                "force_mode": observation_config.calibration_force_mode,
                "min_gripper_gap_m": observation_config.min_gripper_gap_m,
                "max_gripper_gap_m": observation_config.max_gripper_gap_m,
            },
            "normalization": {
                "max_force_n": observation_config.max_force_n,
                "max_force_delta_n": observation_config.max_force_delta_n,
                "max_gripper_gap_m": observation_config.max_gripper_gap_m,
                "max_gripper_gap_delta_m": observation_config.max_gripper_gap_delta_m,
                "max_force_error_integral_ns": observation_config.max_force_error_integral_ns,
                "max_force_derivative_n_per_s": observation_config.max_force_derivative_n_per_s,
            },
            "action_mapping": ["joint5_delta", "desired_force"],
            "control_period_s": env_config.control_period_s,
            "control_steps_per_action": env_config.control_steps_per_action,
            "initial_joint_angles": list(env_config.initial_joint_angles)
            if env_config.initial_joint_angles is not None
            else None,
            "joint5_lift_delta_rad": env_config.joint5_lift_delta_rad,
            "max_gripper_control_delta_m": env_config.max_gripper_control_delta_m,
            "desired_force_min_n": env_config.desired_force_min_n,
            "desired_force_max_n": env_config.desired_force_max_n,
            "joint5_index": env_config.joint5_index,
            "gripper_index": env_config.gripper_index,
        }

    def reset(self) -> np.ndarray:
        self.steps = 0
        self.done = False
        self.last_info = {}
        self.episode_force_sum = 0.0
        self.episode_force_samples = 0
        self.episode_peak_force = 0.0
        self._max_force_drop_n = 0.0
        self._max_relative_force_drop = 0.0
        self._sustained_force_drop_n = 0.0
        self._joint5_lift_progress_rad = 0.0
        self._lifting_this_step = False
        self._left_force_delta_buffer.clear()
        self._left_force_delta_buffer.extend((0.0, 0.0, 0.0))
        self._right_force_delta_buffer.clear()
        self._right_force_delta_buffer.extend((0.0, 0.0, 0.0))
        self._is_slipping_state = False
        self._slip_recovered_this_step = False
        self._slip_recovery_steps = 0
        self._slip_detection_armed = True
        self._slip_rearm_steps = 0
        self._initial_joint5_position = None
        self._episode_max_lift_progress = 0.0
        self._last_recovery_bonus_progress = 0.0
        self._upward_action_magnitude = 0.0
        self._sensor_offset = self._sample_offset(self.observation_config.sensor_offset_range)
        self._motor_offset = self._sample_offset(self.observation_config.motor_offset_range)
        actual_start = self.controller.get_joint_angles()
        requested_start = self.env_config.initial_joint_angles
        if requested_start is None:
            requested_start = tuple(actual_start)
        if getattr(self.controller, "is_physical", False):
            self._validate_physical_start_pose(actual_start, requested_start)
            start_position = list(actual_start)
        else:
            start_position = list(requested_start)
        self.controller.set_initial_position(start_position)
        self._run_controller_steps(self.env_config.settle_steps)

        calibration_info = self._calibrate_stiffness()

        self._initial_object_height = self._obj_height()
        pos = self._obj_height()
        self._initial_object_xy = None if pos is None else np.asarray(self.controller.sim.data.get_body_xpos(self.env_config.default_body_name), dtype=np.float32)[:2]
        
        self.controller.reset_force_pid()
        left_force, right_force = self._read_forces()
        actual = self.controller.get_joint_angles()
        self._initial_joint5_position = float(
            actual[self.env_config.joint5_index]
        )
        self._previous_left_force = left_force
        self._previous_right_force = right_force
        self._previous_gripper_gap = abs(
            float(actual[self.env_config.gripper_index])
        )
        self._previous_desired_force_n = self.env_config.desired_force_min_n

        self.last_info = {"calibration": calibration_info}
        return self.observe()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.done:
            return self.observe(), 0.0, True, {"already_done": True, **self.last_info}


        command = self._apply_action(self._current_command(), action)
        safety_reason = self._run_force_control_steps(
            command,
            self._previous_desired_force_n,
            self._control_steps_per_action(),
        )
        
        self.steps += 1
        reward, terminated, reward_info = self._reward()
        if safety_reason is not None:
            if not terminated:
                reward += self.reward_config.failure_penalty
            terminated = True
            reward_info["reason"] = safety_reason
            reward_info["success"] = False
        truncated = self.steps >= self.env_config.max_steps and not terminated
        self.done = terminated or truncated
        info = {
            "action": action,
            "terminated": terminated,
            "truncated": truncated,
            **reward_info,
        }
        self.last_info = info
        return self.observe(), reward, self.done, info

    def observe(self) -> np.ndarray:
        raw = self.observation_dict()
        values = np.array([raw[name] for name in self.observation_names], dtype=np.float32)
        if not self.observation_config.normalize:
            self._commit_observation_history(raw)
            return values

        normalized = np.array(
            [
                self._normalize_positive(raw["left_force_n"], self.observation_config.max_force_n),
                self._normalize_positive(raw["right_force_n"], self.observation_config.max_force_n),
                self._normalize_symmetric(
                    raw["left_force_delta_n"],
                    self.observation_config.max_force_delta_n,
                ),
                self._normalize_symmetric(
                    raw["right_force_delta_n"],
                    self.observation_config.max_force_delta_n,
                ),
                self._normalize_positive(raw["gripper_gap_m"], self.observation_config.max_gripper_gap_m),
                self._normalize_symmetric(
                    raw["gripper_gap_delta_m"],
                    self.observation_config.max_gripper_gap_delta_m,
                ),
                self._normalize_joint5(raw["joint5_position"]),
                self._normalize_symmetric(
                    raw["force_error_integral_ns"],
                    self.observation_config.max_force_error_integral_ns,
                ),
                self._normalize_symmetric(
                    raw["force_derivative_filtered_n_per_s"],
                    self.observation_config.max_force_derivative_n_per_s,
                ),
                1.0 if raw["is_slipping"] else -1.0,
            ],
            dtype=np.float32,
        )

        normalized[:2] += self._sensor_offset
        normalized[[4, 6]] += self._motor_offset
        if self.observation_config.sensor_noise_std > 0:
            normalized[:2] += self.rng.normal(
                0.0,
                self.observation_config.sensor_noise_std,
                size=2,
            )
        self._commit_observation_history(raw)
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def observation_dict(self) -> Dict[str, float]:
        left_force, right_force = self._read_forces()
        #print(left_force - self._previous_left_force)
        actual = self.controller.get_joint_angles()
        gripper_gap = abs(float(actual[self.env_config.gripper_index]))
        return {
            "left_force_n": left_force,
            "right_force_n": right_force,
            "left_force_delta_n": left_force - self._previous_left_force,
            "right_force_delta_n": right_force - self._previous_right_force,
            "gripper_gap_m": gripper_gap,
            "gripper_gap_delta_m": gripper_gap - self._previous_gripper_gap,
            "joint5_position": float(actual[self.env_config.joint5_index]),
            "force_error_integral_ns": self.controller.force_error_integral,
            "force_derivative_filtered_n_per_s": self.controller.force_derivative_filtered,
            "is_slipping": float(self._is_slipping_state),
        }

    def _commit_observation_history(self, raw: Dict[str, float]) -> None:
        self._previous_left_force = raw["left_force_n"]
        self._previous_right_force = raw["right_force_n"]
        self._previous_gripper_gap = raw["gripper_gap_m"]

    def _calibrate_stiffness(self) -> Dict[str, Any]:
        command = self._current_command()
        reached_threshold = False
        left_force = 0.0
        right_force = 0.0
        calibration_steps = 0
        cfg = self.env_config

        for calibration_steps in range(
            self.observation_config.calibration_max_steps + 1
        ):
            left_force, right_force = self._read_forces()
            force_limit_reason = self._force_limit_reason(left_force, right_force)
            if force_limit_reason is not None:
                self._trigger_safety_stop()
                raise RuntimeError(
                    f"Force safety limit reached during calibration: {force_limit_reason}"
                )
            if self._calibration_force_met(left_force, right_force):
                reached_threshold = True
                break
            if calibration_steps >= self.observation_config.calibration_max_steps:
                raise RuntimeError("Max calibration steps reached")

            command[cfg.gripper_index] -= cfg.calibration_gripper_delta
            self.controller.send_joint_angle_cmd(command)
            self._run_controller_steps(1)

           

        left_force, right_force = self._read_forces()
        gap = max(self._gripper_gap(self._current_command()), self.observation_config.gripper_gap_epsilon_m)
        force_for_stiffness = (
            self.observation_config.min_force_threshold_n
            if reached_threshold
            else max((left_force + right_force) * 0.5, 0.0)
        )
        self.stiffness_n_per_m = force_for_stiffness / gap

        if reached_threshold:
            command[cfg.gripper_index] += cfg.calibration_gripper_delta
            self.controller.send_joint_angle_cmd(command)
            self._run_controller_steps(1)

        return {
            "reached_threshold": reached_threshold,
            "steps": calibration_steps,
            "left_force_n": left_force,
            "right_force_n": right_force,
            "gripper_gap_m": gap,
            "stiffness_n_per_m": self.stiffness_n_per_m,
        }

    def _reward(self) -> Tuple[float, bool, Dict[str, Any]]:
        cfg = self.reward_config
        step_penalty = cfg.step_penalty
        if step_penalty is None:
            step_penalty = -1.0 / float(self.env_config.max_steps)

        left_force, right_force = self._read_forces()
        average_force = 0.5 * (left_force + right_force)
        

        reward = float(step_penalty)

       
        terminated = False
        reason = None

        for sensor_name, force in (("left", left_force), ("right", right_force)):
            if force >= cfg.terminate_force_n:
                reward += cfg.failure_penalty
                terminated = True
                reason = f"{sensor_name}_force_limit"
            #elif cfg.desired_force_min_n <= force < cfg.desired_force_max_n:
            #    reward += cfg.force_reward
            elif cfg.desired_force_max_n <= force < cfg.terminate_force_n:
                reward += cfg.force_penalty

        is_slipping = self._is_slip()
        force_decrease_reward = self._compute_force_decrease_reward(
            cfg,
            left_force,
            right_force,
            is_slipping,
        )
        reward += force_decrease_reward
        reward += self._compute_height_reward(cfg, is_slipping=is_slipping)

        success = False
        if not terminated and self._is_success(left_force, right_force):


            reward += cfg.success_reward #- force_cost
            terminated = True
            success = True
            reason = "success"

        if reason is None:
            reason = "episode step limit exceeded"
        
        return reward, terminated, {
            "reason": reason,
            "success": success,
            "left_force_n": left_force,
            "right_force_n": right_force,
            "episode_peak_force_n": self.episode_peak_force,
            "episode_average_force_n": None,
            "max_force_drop_n": self._max_force_drop_n,
            "max_relative_force_drop": self._max_relative_force_drop,
            "sustained_force_drop_n": self._sustained_force_drop_n,
            "joint5_lift_progress_rad": self._joint5_lift_progress_rad,
            "minimum_force_change_n": None,
            "contact_confidence": None,
            "object_lift_height_m": None,
            "stiffness_n_per_m": self.stiffness_n_per_m,
            "is_slipping": self._is_slipping_state,
            "force_decrease_reward": force_decrease_reward,
            "slip_recovered": self._slip_recovered_this_step,
            "left_force_delta_window_n": sum(self._left_force_delta_buffer),
            "right_force_delta_window_n": sum(self._right_force_delta_buffer),
            "upward_action_magnitude": self._upward_action_magnitude,
        }

    def _is_success(self, left_force: float, right_force: float) -> bool:
        cfg = self.reward_config
        average_force = 0.5 * (left_force + right_force)
        if self.controller.ground_truth_pos is False:
            return (self.controller.get_joint_angles()[4] < cfg.success_lift_angle
                     and average_force > cfg.desired_force_min_n 
                     and left_force < cfg.desired_force_max_n 
                     and right_force < cfg.desired_force_max_n)

        return (
            self._obj_height() - self._initial_object_height > cfg.success_lift_height_m
            and average_force >= cfg.desired_force_min_n
            and left_force < cfg.desired_force_max_n
            and right_force < cfg.desired_force_max_n
        )

    def _compute_force_decrease_reward(
        self,
        cfg: RewardConfig,
        left_force: float,
        right_force: float,
        is_slipping: bool,
    ) -> float:
        if is_slipping:
            return 0.0

        previous_average_force = 0.5 * (
            self._previous_left_force + self._previous_right_force
        )
        average_force = 0.5 * (left_force + right_force)
        force_decrease = max(0.0, previous_average_force - average_force)
        rewarded_force_decrease = min(
            force_decrease,
            max(0.0, cfg.max_rewarded_force_change_n),
        )
        return cfg.force_change_reward_coef * rewarded_force_decrease

    def _compute_height_reward(
        self,
        cfg: RewardConfig,
        is_slipping: Optional[bool] = None,
    ) -> float:
        rew = 0
        if is_slipping is None:
            is_slipping = self._is_slip()
        if is_slipping and self._lifting_this_step:
            rew += cfg.slip_lift_penalty * self._upward_action_magnitude
        rew += self._compute_slip_recovery_bonus(cfg)

        if self.controller.ground_truth_pos == True:
            current_height = self.controller.sim.data.get_body_xpos(self.env_config.default_body_name)[2]
            delta_height = current_height - self._initial_object_height
            if delta_height > cfg.lift_reward_height_m:
                print("lift rew")
                rew += cfg.lift_reward
            if delta_height > cfg.high_lift_reward_height_m:
                print("high lift rew")

                rew += cfg.high_lift_reward
            return rew
        af = self.controller.get_force_average()
        current_j5_angle = self.controller.get_joint_angles()[4]
        #neg is higher angle
        if af > cfg.desired_force_min_n and not is_slipping:
            y = self._f_height_reward(current_j5_angle)
            rew+= y

        return rew

    def _is_slip(self) -> bool:
        left_force, right_force = self._read_forces()
        self._left_force_delta_buffer.append(
            left_force - self._previous_left_force
        )
        self._right_force_delta_buffer.append(
            right_force - self._previous_right_force
        )

        cfg = self.observation_config
        left_window_delta = sum(self._left_force_delta_buffer)
        right_window_delta = sum(self._right_force_delta_buffer)
        slip_detected = (
            left_window_delta < -cfg.slip_drop_threshold_n
            or right_window_delta < -cfg.slip_drop_threshold_n
        )
        no_new_drop = (
            left_window_delta >= -cfg.slip_recovery_drop_tolerance_n
            and right_window_delta >= -cfg.slip_recovery_drop_tolerance_n
        )
        bilateral_contact = (
            left_force >= cfg.min_force_threshold_n
            and right_force >= cfg.min_force_threshold_n
        )
        self._slip_recovered_this_step = False
        if self._is_slipping_state:
            if slip_detected:
                self._slip_recovery_steps = 0
                return True

            recovery_force_threshold = max(
                cfg.min_force_threshold_n,
                cfg.slip_recovery_force_ratio * self._previous_desired_force_n,
            )
            bilateral_force_rebuilt = (
                left_force >= recovery_force_threshold
                and right_force >= recovery_force_threshold
            )

            if bilateral_force_rebuilt and no_new_drop:
                self._slip_recovery_steps += 1
            else:
                self._slip_recovery_steps = 0

            confirmation_steps = max(1, int(cfg.slip_recovery_confirmation_steps))
            if self._slip_recovery_steps >= confirmation_steps:
                self._is_slipping_state = False
                self._slip_recovered_this_step = True
                self._slip_recovery_steps = 0
                self._slip_detection_armed = False
                self._slip_rearm_steps = 0
        elif self._slip_detection_armed:
            if slip_detected:
                self._is_slipping_state = True
                self._slip_recovery_steps = 0
        else:
            if bilateral_contact and no_new_drop:
                self._slip_rearm_steps += 1
            else:
                self._slip_rearm_steps = 0

            rearm_steps = max(1, int(cfg.slip_rearm_confirmation_steps))
            if self._slip_rearm_steps >= rearm_steps:
                self._slip_detection_armed = True
                self._slip_rearm_steps = 0
        return self._is_slipping_state

    def _compute_slip_recovery_bonus(self, cfg: RewardConfig) -> float:
        progress = self._normalized_lift_progress(cfg)
        self._episode_max_lift_progress = max(
            self._episode_max_lift_progress,
            progress,
        )
        if not self._slip_recovered_this_step:
            return 0.0

        required_progress = (
            self._last_recovery_bonus_progress
            + max(0.0, cfg.slip_recovery_bonus_min_lift_progress)
        )
        if self._episode_max_lift_progress < required_progress:
            return 0.0

        self._last_recovery_bonus_progress = self._episode_max_lift_progress
        return cfg.slip_recovery_bonus

    def _normalized_lift_progress(self, cfg: RewardConfig) -> float:
        if self.controller.ground_truth_pos is True:
            if self._initial_object_height is None or cfg.success_lift_height_m <= 0.0:
                return 0.0
            return max(
                0.0,
                (self._obj_height() - self._initial_object_height)
                / cfg.success_lift_height_m,
            )

        initial_joint5 = self._initial_joint5_position
        if initial_joint5 is None:
            configured_start = self.env_config.initial_joint_angles
            if configured_start is None:
                return 0.0
            initial_joint5 = float(configured_start[self.env_config.joint5_index])

        success_distance = initial_joint5 - cfg.success_lift_angle
        if success_distance <= 0.0:
            return 0.0
        current_joint5 = float(
            self.controller.get_joint_angles()[self.env_config.joint5_index]
        )
        return max(0.0, (initial_joint5 - current_joint5) / success_distance)

    def _calibration_force_met(self, left_force: float, right_force: float) -> bool:
        
        threshold = self.observation_config.min_force_threshold_n
        mode = self.observation_config.calibration_force_mode
        if mode == "both":
            return left_force >= threshold and right_force >= threshold
        if mode == "either":
            return left_force >= threshold or right_force >= threshold
        if mode != "average":
            raise ValueError("calibration_force_mode must be 'average', 'both', or 'either'")
        return 0.5 * (left_force + right_force) >= threshold

    def _apply_action(self, command: Sequence[float], action: np.ndarray) -> list[float]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.num_actions,):
            raise ValueError(
                f"Expected action shape ({self.num_actions},), got {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            self._trigger_safety_stop()
            raise RuntimeError("Policy returned a non-finite action")
        if np.any(action < -1.000001) or np.any(action > 1.000001):
            self._trigger_safety_stop()
            raise RuntimeError(f"Policy action was outside [-1, 1]: {action.tolist()}")
        action = np.clip(action, -1.0, 1.0)
        next_command = list(command)
        cfg = self.env_config

        joint5_position = float(command[cfg.joint5_index])
        requested_joint5 = (
            joint5_position + action[0] * cfg.joint5_lift_delta_rad
        )
        self._lifting_this_step = requested_joint5 < joint5_position
        self._upward_action_magnitude = max(float(action[0]), 0.0)
        next_command[cfg.joint5_index] = requested_joint5

        desired_force = cfg.desired_force_min_n + (
        (action[1] + 1.0) / 2.0
        * (cfg.desired_force_max_n - cfg.desired_force_min_n))
        self._previous_desired_force_n = float(desired_force)
        for idx in (cfg.joint5_index, cfg.gripper_index):
        
            next_command[idx] = self._clamp_joint(idx, next_command[idx])
        return next_command

    def _current_command(self) -> list[float]:
        try:
            command = self.controller.get_joint_angle_cmd()
            if command:
                return list(command)
        except Exception:
            pass
        return list(self.controller.get_joint_angles())

    def _clamp_joint(self, joint_index: int, value: float) -> float:
        bounds = getattr(self.controller, "joint_bounds", None)
        if joint_index == self.env_config.gripper_index:
            lo = self.observation_config.min_gripper_gap_m
            hi = self.observation_config.max_gripper_gap_m
            if bounds and len(bounds) >= (joint_index * 2 + 2):
                lo = max(lo, float(bounds[joint_index * 2]))
                hi = min(hi, float(bounds[joint_index * 2 + 1]))
            if hi < lo:
                raise ValueError("Gripper minimum gap exceeds its maximum gap")
            return max(lo, min(float(value), hi))

        if bounds and len(bounds) >= (joint_index * 2 + 2):
            lo = float(bounds[joint_index * 2])
            hi = float(bounds[joint_index * 2 + 1])
            return max(lo, min(float(value), hi))
        if joint_index == self.env_config.joint5_index:
            return max(-1.22, min(float(value), 1.22))
        return float(value)

    def _gripper_gap(self, command: Sequence[float]) -> float:
        return abs(float(command[self.env_config.gripper_index]))

    def _read_forces(self) -> Tuple[float, float]:
        left = float(self.controller.get_force_left())
        right = float(self.controller.get_force_right())
        if not math.isfinite(left) or not math.isfinite(right):
            self._trigger_safety_stop()
            raise RuntimeError(f"Non-finite force feedback: left={left}, right={right}")
        return max(0.0, left), max(0.0, right)

    def _obj_height(self) -> float:
        if self.controller.ground_truth_pos == True:
            return self.controller.sim.data.get_body_xpos(self.env_config.default_body_name)[2]
        return None

  

    def _object_out_of_bounds(self) -> bool:
        current_pos = self._obj_height()
        if current_pos is None or self._initial_object_xy is None:
            return False
        distance = np.linalg.norm(np.asarray(current_pos[:2], dtype=np.float32) - self._initial_object_xy)
        return bool(distance > self.reward_config.out_of_bounds_radius_m)

    def _run_controller_steps(self, steps: int) -> None:
        for _ in range(max(0, int(steps))):
            self.controller.step()

    def _run_force_control_steps(
        self,
        command: Sequence[float],
        desired_force: float,
        steps: int,
    ) -> Optional[str]:
        current_command = list(command)
        step_count = max(0, int(steps))
        self._max_force_drop_n = 0.0
        self._max_relative_force_drop = 0.0
        self._sustained_force_drop_n = 0.0
        self._joint5_lift_progress_rad = 0.0
        if step_count == 0:
            self.controller.send_joint_angle_cmd(current_command)
            return None
        print(desired_force)
        gripper_index = self.env_config.gripper_index
        joint5_index = self.env_config.joint5_index
        previous_left_force, previous_right_force = self._read_forces()
        initial_min_force = min(previous_left_force, previous_right_force)
        initial_joint5_position = float(
            self.controller.get_joint_angles()[joint5_index]
        )
        for _ in range(step_count):
            left_force, right_force = self._read_forces()
            force_limit_reason = self._force_limit_reason(left_force, right_force)
            if force_limit_reason is not None:
                self._trigger_safety_stop()
                return force_limit_reason
            gripper_delta = self.controller.pid(desired_force, self.controller.dt)
            gripper_delta = float(
                np.clip(
                    gripper_delta,
                    -self.env_config.max_gripper_control_delta_m,
                    self.env_config.max_gripper_control_delta_m,
                )
            )
            current_command[gripper_index] = self._clamp_joint(
                gripper_index,
                current_command[gripper_index] - gripper_delta,
            )
            self.controller.send_joint_angle_cmd(current_command)
            self.controller.step()

            left_force, right_force = self._read_forces()
            force_limit_reason = self._force_limit_reason(left_force, right_force)
            if force_limit_reason is not None:
                self._trigger_safety_stop()
                return force_limit_reason
            left_drop = max(0.0, previous_left_force - left_force)
            right_drop = max(0.0, previous_right_force - right_force)
            force_drop = max(left_drop, right_drop)
            relative_force_drop = max(
                left_drop
                / max(
                    previous_left_force,
                    self.env_config.relative_force_drop_epsilon_n,
                ),
                right_drop
                / max(
                    previous_right_force,
                    self.env_config.relative_force_drop_epsilon_n,
                ),
            )
            self._max_force_drop_n = max(self._max_force_drop_n, force_drop)
            self._max_relative_force_drop = max(
                self._max_relative_force_drop,
                relative_force_drop,
            )

            previous_left_force = left_force
            previous_right_force = right_force

        final_min_force = min(previous_left_force, previous_right_force)
        self._sustained_force_drop_n = max(
            0.0,
            initial_min_force - final_min_force,
        )
        actual = self.controller.get_joint_angles()
        self._joint5_lift_progress_rad = (
            initial_joint5_position - float(actual[joint5_index])
        )
        return None

    def _control_steps_per_action(self) -> int:
        configured = self.env_config.control_steps_per_action
        if configured is not None:
            return max(1, int(configured))
        dt = float(self.controller.dt)
        if dt <= 0.0:
            raise ValueError("Controller dt must be positive")
        return max(1, int(round(self.env_config.control_period_s / dt)))

    def _force_limit_reason(self, left_force: float, right_force: float) -> Optional[str]:
        limit = self.reward_config.terminate_force_n
        if left_force >= limit:
            return "left_force_limit"
        if right_force >= limit:
            return "right_force_limit"
        return None

    def _trigger_safety_stop(self) -> None:
        if not getattr(self.controller, "is_physical", False):
            return
        stop = getattr(self.controller, "emergency_stop", None)
        if stop is None:
            stop = getattr(self.controller, "stop", None)
        if stop is not None:
            stop()

    def _validate_physical_start_pose(
        self,
        actual: Sequence[float],
        expected: Sequence[float],
    ) -> None:
        if len(actual) != 7 or len(expected) != 7:
            raise ValueError("Physical and expected start poses must contain seven values")
        if not all(math.isfinite(float(value)) for value in (*actual, *expected)):
            raise RuntimeError("Physical start pose contains a non-finite value")
        joint5_index = self.env_config.joint5_index
        gripper_index = self.env_config.gripper_index
        joint5_error = abs(
            float(actual[joint5_index]) - float(expected[joint5_index])
        )
        gripper_error = abs(
            float(actual[gripper_index]) - float(expected[gripper_index])
        )
        if (
            joint5_error > self.env_config.physical_start_pose_tolerance_rad
            or gripper_error > self.env_config.physical_start_gripper_tolerance_m
        ):
            raise RuntimeError(
                "Physical robot is not staged at the trained start pose. Move it with a "
                "verified low-speed trajectory before running the policy. "
                f"joint5_error={joint5_error:.6f} rad, "
                f"gripper_error={gripper_error:.6f} m"
            )

    def _normalize_positive(self, value: float, maximum: float) -> float:
        if maximum <= 0:
            return 0.0
        clipped = max(0.0, min(float(value), maximum))
        return 2.0 * (clipped / maximum) - 1.0

    def _normalize_symmetric(self, value: float, maximum: float) -> float:
        if maximum <= 0:
            return 0.0
        return max(-1.0, min(float(value) / maximum, 1.0))

    def _normalize_joint5(self, value: float) -> float:
        lo = self._clamp_joint(self.env_config.joint5_index, -1.22)
        hi = self._clamp_joint(self.env_config.joint5_index, 1.22)
        if hi <= lo:
            return 0.0
        clipped = max(lo, min(float(value), hi))
        return 2.0 * ((clipped - lo) / (hi - lo)) - 1.0

    def _sample_offset(self, offset_range: float) -> float:
        if offset_range <= 0:
            return 0.0
        return float(self.rng.uniform(-offset_range, offset_range))

    def _clamp(value, min_val, max_val):
        return max(min_val, min(value, max_val))

    

    A = 2.2e-6
    B = 3e-3
    X_ZERO = 6.6 - math.log(B / A)

    def _f_height_reward(self, x: float) -> float:
        return self.B * math.expm1(self.X_ZERO - x)
