from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
from controllers.controller import Controller

import numpy as np



@dataclass
class ObservationConfig:
    min_force_threshold_n: float = 0.00001
    calibration_max_steps: int = 200
    calibration_force_mode: str = "either"
    normalize: bool = True
    max_force_n: float = 5.0
    max_stiffness_n_per_m: float = 1000.0
    max_gripper_gap_m: float = 0.07
    gripper_gap_epsilon_m: float = 1e-4
    sensor_offset_range: float = 0.0
    sensor_noise_std: float = 0.0
    motor_offset_range: float = 0.0



@dataclass
class RewardConfig:
    desired_force_min_n: float = 0.1
    desired_force_max_n: float = 4.0
    terminate_force_n: float = 5.0
    force_reward: float = 0.0008
    force_penalty: float = -0.0025
    lift_reward_height_m: float = 0.005
    lift_reward: float = 0.0008
    high_lift_reward_height_m: float = 0.015
    high_lift_reward: float = 0.0016
    success_lift_height_m: float = 0.040
    success_reward: float = 1.0
    failure_penalty: float = -1.0
    step_penalty: Optional[float] = None
    out_of_bounds_radius_m: float = 0.25

    min_force_reward_coef: float = 1.5
    ideal_force_mass_ratio = 0.15 / 0.05
    ratio_tolerance = 0.1

    low_lift_angle: float = -0.80
    high_lift_angle: float = -0.86
    success_lift_angle: float = -0.98




@dataclass
class GraspEnvConfig:
    initial_joint_angles: Tuple[float, ...] = (0, 1.5, -0.3, 0, -0.7, 0, 0.02, -0.02) #start w open gripper and close via calibration ONLY
    max_steps: int = 250
    settle_steps: int = 5
    control_steps_per_action: int = 4
    gripper_delta_m: float = 0.0001
    joint5_lift_delta_rad: float = -0.01
    joint5_lower_delta_rad: float = 0.01
    calibration_gripper_delta: float = 0.00001

    
    joint5_index: int = 4
    left_gripper_index: int = 6
    right_gripper_index: int = 7
    default_body_name: str = "grasp_box"


ObjectHeightFn = Callable[[Any], Optional[float]]
ObjectPositionFn = Callable[[Any], Optional[Sequence[float]]]


class GraspPPOEnv:
    """Small controller-backed grasping environment for discrete PPO.

    The environment deliberately avoids a Gym dependency. It exposes a familiar
    reset/step interface and keeps robot-specific assumptions configurable.
    """

    num_actions = 2 #continuous action vector of [gripper delta, joint5 delta]
    observation_names = (
        "left_force_n",
        "right_force_n",
        "stiffness_n_per_m",
        "gripper_gap_m",
        "joint5_position",
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

        self.episode_force_sum = 0.0
        self.episode_force_samples = 0
        self.sample_force = False

        
    @property
    def observation_dim(self) -> int:
        return len(self.observation_names)

    def reset(self) -> np.ndarray:
        self.steps = 0
        self.done = False
        self.last_info = {}
        self.episode_force_sum = 0.0
        self.episode_force_samples = 0
        self.sample_force = False
        self._sensor_offset = self._sample_offset(self.observation_config.sensor_offset_range)
        self._motor_offset = self._sample_offset(self.observation_config.motor_offset_range)
        if self.env_config.initial_joint_angles is None:
            self.env_config.initial_joint_angles = self.controller.get_joint_angles()
        self.controller.set_initial_position(list(self.env_config.initial_joint_angles))


        #self._run_controller_steps(self.env_config.settle_steps)
        
        
        calibration_info = self._calibrate_stiffness()

        self._initial_object_height = self._obj_height()
        pos = self._obj_height()
        self._initial_object_xy = None if pos is None else np.asarray(self.controller.sim.data.get_body_xpos(self.env_config.default_body_name), dtype=np.float32)[:2]
        

        self.last_info = {"calibration": calibration_info}
        return self.observe()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.done:
            return self.observe(), 0.0, True, {"already_done": True, **self.last_info}


        command = self._apply_action(self._current_command(), action)
        self.controller.send_joint_angle_cmd(command)
        self._run_controller_steps(self.env_config.control_steps_per_action)
        
        self.steps += 1
        reward, terminated, reward_info = self._reward()
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
            return values

        normalized = np.array(
            [
                self._normalize_positive(raw["left_force_n"], self.observation_config.max_force_n),
                self._normalize_positive(raw["right_force_n"], self.observation_config.max_force_n),
                self._normalize_positive(
                    raw["stiffness_n_per_m"],
                    self.observation_config.max_stiffness_n_per_m,
                ),
                self._normalize_positive(raw["gripper_gap_m"], self.observation_config.max_gripper_gap_m),
                self._normalize_joint5(raw["joint5_position"]),
            ],
            dtype=np.float32,
        )

        normalized[:3] += self._sensor_offset
        normalized[3:] += self._motor_offset
        if self.observation_config.sensor_noise_std > 0:
            normalized[:3] += self.rng.normal(0.0, self.observation_config.sensor_noise_std, size=3)
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def observation_dict(self) -> Dict[str, float]:
        left_force, right_force = self._read_forces()
        command = self._current_command()
        return {
            "left_force_n": left_force,
            "right_force_n": right_force,
            "stiffness_n_per_m": self.stiffness_n_per_m,
            "gripper_gap_m": self._gripper_gap(command),
            "joint5_position": float(command[self.env_config.joint5_index]),
        }

    def _calibrate_stiffness(self) -> Dict[str, Any]:
        command = self._current_command()
        reached_threshold = False
        left_force = 0.0
        right_force = 0.0

        while True:


            left_force = self.controller.get_force_left()
            right_force = self.controller.get_force_right()
            if left_force != 0.0 or right_force != 0.0:
                #print("left " + str(left_force) + " right " + str(right_force))
                break
   
            command[6] -= self.env_config.calibration_gripper_delta
            command[7] += self.env_config.calibration_gripper_delta
            self.controller.send_joint_angle_cmd(command)
            self._run_controller_steps(1)
        
        
        reached_threshold = True
        left_force, right_force = self._read_forces()
        gap = max(self._gripper_gap(self._current_command()), self.observation_config.gripper_gap_epsilon_m)
        force_for_stiffness = (
            self.observation_config.min_force_threshold_n
            if reached_threshold
            else max((left_force + right_force) * 0.5, 0.0)
        )
        self.stiffness_n_per_m = force_for_stiffness / gap

        #
        command[6] += self.env_config.calibration_gripper_delta
        command[7] -= self.env_config.calibration_gripper_delta
        self.controller.send_joint_angle_cmd(command)
        self._run_controller_steps(1)

        return {
            "reached_threshold": reached_threshold,
            "steps": 0,
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
        reward = float(step_penalty)
        reward += cfg.force_penalty*((left_force+right_force)/2)
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
        
        reward += self._compute_height_reward(cfg)

        if self.sample_force:
            self.episode_force_sum += 0.5 * (left_force + right_force)
            self.episode_force_samples += 1
        success = False
        if not terminated and self._is_success(left_force, right_force):
            print("joint 5 success angle: " + str(self.controller.get_joint_angles()[4]))
            average_force = (
                self.episode_force_sum / self.episode_force_samples
                if self.episode_force_samples
                else 0.5 * (left_force + right_force)
            )
            force_range = max(cfg.desired_force_max_n - cfg.desired_force_min_n, 1e-8)
            normalized_force_penalty = np.clip(
                (average_force - cfg.desired_force_min_n) / force_range,
                0.0,
                1.0,
            )
            
            reward += cfg.success_reward - cfg.min_force_reward_coef * normalized_force_penalty
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
            "object_lift_height_m": None,
            "stiffness_n_per_m": self.stiffness_n_per_m,
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

    def _compute_height_reward(self, cfg: RewardConfig) -> float:
        rew = 0
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
        if current_j5_angle < cfg.low_lift_angle and af> cfg.desired_force_min_n :
            rew += cfg.lift_reward  
            print("lift rew")
        if current_j5_angle < cfg.high_lift_angle and af > cfg.desired_force_min_n:
            rew += cfg.high_lift_reward
            print("high lift rew")
        return rew


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
        next_command = list(command)
        cfg = self.env_config

        next_command[4] += action[0] * cfg.joint5_lift_delta_rad
        next_command[6] += action[1] * cfg.gripper_delta_m
        next_command[7] -= action[1] * cfg.gripper_delta_m #TODO: individual gripper movement?

        for idx in (cfg.joint5_index, cfg.left_gripper_index, cfg.right_gripper_index):
        
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
        if bounds and len(bounds) >= (joint_index * 2 + 2):
            lo = float(bounds[joint_index * 2])
            hi = float(bounds[joint_index * 2 + 1])
            return max(lo, min(float(value), hi))
        if joint_index == self.env_config.left_gripper_index:
            return max(0.0, min(float(value), 0.035))
        if joint_index == self.env_config.right_gripper_index:
            return max(-0.035, min(float(value), 0.0))
        if joint_index == self.env_config.joint5_index:
            return max(-1.22, min(float(value), 1.22))
        return float(value)

    def _gripper_gap(self, command: Sequence[float]) -> float:
        return abs(
            float(command[self.env_config.left_gripper_index])
            - float(command[self.env_config.right_gripper_index])
        )

    def _read_forces(self) -> Tuple[float, float]:
        return max(0.0, float(self.controller.get_force_left())), max(
            0.0,
            float(self.controller.get_force_right()),
        )

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

    def _normalize_positive(self, value: float, maximum: float) -> float:
        if maximum <= 0:
            return 0.0
        clipped = max(0.0, min(float(value), maximum))
        return 2.0 * (clipped / maximum) - 1.0

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
