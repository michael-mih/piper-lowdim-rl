from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from controllers.sim_controller import SimController
from learning.env import GraspEnvConfig, GraspPPOEnv, ObservationConfig, RewardConfig
from learning.ppo import (
    PPOAgent,
    PPOConfig,
    PPOTrainer,
    ParallelPPOTrainer,
    SubprocessVectorEnv,
)
from scripts.build_sim import combined_xml


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, dt: float = 0.01) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral_error = 0.0
        self.prev_error = 0.0

    def calculate(self, target: float, current: float) -> float:
        error = target - current
        self.integral_error += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral_error + self.kd * derivative


@dataclass(frozen=True)
class TrainingBoxSpec:
    name: str
    half_extents_m: Tuple[float, float, float]
    mass_kg: float
    rgba: Tuple[float, float, float, float]

    @property
    def body_name(self) -> str:
        return f"grasp_box_{self.name}"

    @property
    def geom_name(self) -> str:
        return f"{self.body_name}_geom"


@dataclass(frozen=True)
class TrainingScene:
    path: Path
    support_xy: Tuple[float, float]
    support_top_z: float


@dataclass(frozen=True)
class BoxSample:
    half_extents_m: Tuple[float, float, float]
    mass_kg: float
    rgba: Tuple[float, float, float, float]


@dataclass(frozen=True)
class BoxDistribution:
    x_half_extent_range_m: Tuple[float, float] = (0.016, 0.024)
    y_half_extent_range_m: Tuple[float, float] = (0.008, 0.014)
    z_half_extent_range_m: Tuple[float, float] = (0.036, 0.044)
    mass_range_kg: Tuple[float, float] = (0.1, 0.6)

    def sample(self, rng: np.random.Generator) -> BoxSample:
        half_extents = (
            _sample_range(rng, self.x_half_extent_range_m),
            _sample_range(rng, self.y_half_extent_range_m),
            _sample_range(rng, self.z_half_extent_range_m),
        )
        mass = _sample_range(rng, self.mass_range_kg)
        return BoxSample(
            half_extents_m=half_extents,
            mass_kg=mass,
            rgba=_mass_color(mass, self.mass_range_kg),
        )


TRAINING_BOXES: Tuple[TrainingBoxSpec, ...] = (
    TrainingBoxSpec("red", (0.020, 0.010, 0.040), 0.10, (0.90, 0.15, 0.08, 1.0)),
    TrainingBoxSpec("green", (0.018, 0.012, 0.042), 0.08, (0.10, 0.65, 0.22, 1.0)),
    TrainingBoxSpec("blue", (0.022, 0.009, 0.038), 0.12, (0.12, 0.30, 0.90, 1.0)),
)
RANDOMIZED_BOX_TEMPLATE = TrainingBoxSpec(
    "randomized",
    (0.020, 0.010, 0.040),
    0.10,
    (0.10, 0.65, 0.22, 1.0),
)
DEFAULT_BOX_DISTRIBUTION = BoxDistribution()

BOX_FRICTION = "1 0.1 0.01"
PAD_BOX_FRICTION = "1 1 0.1 0.01 0.01"


def _format_floats(values: Sequence[float]) -> str:
    return " ".join(f"{float(value):.6g}" for value in values)


def _parse_floats(value: str) -> Tuple[float, ...]:
    return tuple(float(part) for part in value.split())


def _sample_range(rng: np.random.Generator, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    if hi < lo:
        raise ValueError(f"Invalid range: {bounds}")
    return float(rng.uniform(lo, hi))


def _lerp_color(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    t: float,
) -> Tuple[float, float, float]:
    return tuple(float(start[idx] + (end[idx] - start[idx]) * t) for idx in range(3))


def _mass_color(
    mass_kg: float,
    mass_range_kg: Tuple[float, float],
) -> Tuple[float, float, float, float]:
    lo, hi = mass_range_kg
    t = 0.0 if hi <= lo else float(np.clip((mass_kg - lo) / (hi - lo), 0.0, 1.0))
    light = (0.12, 0.30, 0.90)
    medium = (0.10, 0.65, 0.22)
    heavy = (0.90, 0.15, 0.08)
    if t <= 0.5:
        rgb = _lerp_color(light, medium, t * 2.0)
    else:
        rgb = _lerp_color(medium, heavy, (t - 0.5) * 2.0)
    return (*rgb, 1.0)


def _find_named(root: ET.Element, tag: str, name: str) -> Optional[ET.Element]:
    for element in root.iter(tag):
        if element.get("name") == name:
            return element
    return None


def _remove_existing_boxes(worldbody: ET.Element, box_specs: Sequence[TrainingBoxSpec]) -> None:
    body_names = {"grasp_box", *(box.body_name for box in box_specs)}
    for child in list(worldbody):
        if child.tag == "body" and child.get("name") in body_names:
            worldbody.remove(child)


def _remove_existing_box_contacts(contact: ET.Element, box_specs: Sequence[TrainingBoxSpec]) -> None:
    geom_names = {"grasp_box_geom", *(box.geom_name for box in box_specs)}
    for pair in list(contact):
        if pair.get("geom1") in geom_names or pair.get("geom2") in geom_names:
            contact.remove(pair)


def _inactive_box_xy(index: int, support_xy: Tuple[float, float]) -> Tuple[float, float]:
    y_offsets = (-0.18, 0.18, -0.30, 0.30)
    return support_xy[0] + 0.22, support_xy[1] + y_offsets[(index - 1) % len(y_offsets)]


def build_three_box_scene_xml(
    base_model_path: str,
    box_specs: Sequence[TrainingBoxSpec],
) -> TrainingScene:
    base_path = Path(base_model_path)
    tree = ET.parse(base_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Could not find worldbody in {base_path}")

    support = _find_named(worldbody, "geom", "grasp_box_support")
    if support is None:
        raise ValueError(f"Could not find grasp_box_support in {base_path}")

    support_pos = _parse_floats(support.get("pos", "0 0 0"))
    support_size = _parse_floats(support.get("size", "0 0 0"))
    if len(support_pos) < 3 or len(support_size) < 3:
        raise ValueError("grasp_box_support must define 3D pos and size values")

    support_xy = (support_pos[0], support_pos[1])
    support_top_z = support_pos[2] + support_size[2]

    _remove_existing_boxes(worldbody, (*box_specs, RANDOMIZED_BOX_TEMPLATE))

    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    _remove_existing_box_contacts(contact, (*box_specs, RANDOMIZED_BOX_TEMPLATE))

    for index, box in enumerate(box_specs):
        if index == 0:
            x, y = support_xy
            z = support_top_z + box.half_extents_m[2]
        else:
            x, y = _inactive_box_xy(index, support_xy)
            z = box.half_extents_m[2]

        body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": box.body_name,
                "pos": _format_floats((x, y, z)),
            },
        )
        ET.SubElement(body, "freejoint", {"name": f"{box.body_name}_freejoint"})
        ET.SubElement(
            body,
            "geom",
            {
                "name": box.geom_name,
                "type": "box",
                "size": _format_floats(box.half_extents_m),
                "rgba": _format_floats(box.rgba),
                "mass": f"{box.mass_kg:.6g}",
                "condim": "6",
                "friction": BOX_FRICTION,
            },
        )

    for pad_name in ("left_gripper_pad", "right_gripper_pad"):
        for box in box_specs:
            ET.SubElement(
                contact,
                "pair",
                {
                    "geom1": pad_name,
                    "geom2": box.geom_name,
                    "condim": "6",
                    "friction": PAD_BOX_FRICTION,
                },
            )

    output_path = (
        base_path
        if base_path.stem.endswith("_three_boxes")
        else base_path.with_name(f"{base_path.stem}_three_boxes{base_path.suffix}")
    )
    tree.write(output_path)
    return TrainingScene(
        path=output_path,
        support_xy=support_xy,
        support_top_z=support_top_z,
    )


def build_randomized_box_scene_xml(
    base_model_path: str,
    box_template: TrainingBoxSpec = RANDOMIZED_BOX_TEMPLATE,
) -> TrainingScene:
    base_path = Path(base_model_path)
    tree = ET.parse(base_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Could not find worldbody in {base_path}")

    support = _find_named(worldbody, "geom", "grasp_box_support")
    if support is None:
        raise ValueError(f"Could not find grasp_box_support in {base_path}")

    support_pos = _parse_floats(support.get("pos", "0 0 0"))
    support_size = _parse_floats(support.get("size", "0 0 0"))
    if len(support_pos) < 3 or len(support_size) < 3:
        raise ValueError("grasp_box_support must define 3D pos and size values")

    support_xy = (support_pos[0], support_pos[1])
    support_top_z = support_pos[2] + support_size[2]

    _remove_existing_boxes(worldbody, (*TRAINING_BOXES, box_template))

    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    _remove_existing_box_contacts(contact, (*TRAINING_BOXES, box_template))

    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": box_template.body_name,
            "pos": _format_floats(
                (
                    support_xy[0],
                    support_xy[1],
                    support_top_z + box_template.half_extents_m[2],
                )
            ),
        },
    )
    ET.SubElement(body, "freejoint", {"name": f"{box_template.body_name}_freejoint"})
    ET.SubElement(
        body,
        "geom",
        {
            "name": box_template.geom_name,
            "type": "box",
            "size": _format_floats(box_template.half_extents_m),
            "rgba": _format_floats(box_template.rgba),
            "mass": f"{box_template.mass_kg:.6g}",
            "condim": "6",
            "friction": BOX_FRICTION,
        },
    )

    for pad_name in ("left_gripper_pad", "right_gripper_pad"):
        ET.SubElement(
            contact,
            "pair",
            {
                "geom1": pad_name,
                "geom2": box_template.geom_name,
                "condim": "6",
                "friction": PAD_BOX_FRICTION,
            },
        )

    output_path = (
        base_path
        if base_path.stem.endswith("_randomized_boxes")
        else base_path.with_name(f"{base_path.stem}_randomized_boxes{base_path.suffix}")
    )
    tree.write(output_path)
    return TrainingScene(
        path=output_path,
        support_xy=support_xy,
        support_top_z=support_top_z,
    )


def _box_inertia_diagonal(
    half_extents_m: Tuple[float, float, float],
    mass_kg: float,
) -> Tuple[float, float, float]:
    hx, hy, hz = half_extents_m
    scale = mass_kg / 3.0
    return (
        scale * (hy * hy + hz * hz),
        scale * (hx * hx + hz * hz),
        scale * (hx * hx + hy * hy),
    )


class MultiBoxSimController(SimController):
    def __init__(
        self,
        pid_controllers: Sequence[PIDController],
        model_path: str,
        box_specs: Sequence[TrainingBoxSpec],
        support_xy: Tuple[float, float],
        support_top_z: float,
        render: bool = True,
    ) -> None:
        super().__init__(pid_controllers=pid_controllers, model_path=model_path, render=render)
        self.box_specs = tuple(box_specs)
        self.support_xy = support_xy
        self.support_top_z = support_top_z
        self.active_box_index = 0
        self._box_joint_state = {
            box.body_name: self._free_joint_state_for_body(box.body_name)
            for box in self.box_specs
        }
        self._position_boxes()
        self._passive_joint_state = self._capture_passive_joint_state()

    def set_active_box(self, box_index: int) -> None:
        if box_index < 0 or box_index >= len(self.box_specs):
            raise IndexError(f"Box index {box_index} is out of range")
        self.active_box_index = int(box_index)

    def set_initial_position(self, initial_pos: list[float]) -> None:
        super().set_initial_position(initial_pos)
        self._position_boxes()
        self._passive_joint_state = self._capture_passive_joint_state()

    def _free_joint_state_for_body(self, body_name: str) -> Tuple[int, int]:
        body_id = self.sim.model.body_name2id(body_name)
        joint_id = int(self.sim.model.body_jntadr[body_id])
        if joint_id < 0:
            raise ValueError(f"{body_name} must have a free joint")
        qpos_len, qvel_len = self._joint_state_lengths(joint_id)
        if qpos_len != 7 or qvel_len != 6:
            raise ValueError(f"{body_name} must have a free joint")
        return (
            int(self.sim.model.jnt_qposadr[joint_id]),
            int(self.sim.model.jnt_dofadr[joint_id]),
        )

    def _box_initial_position(
        self,
        index: int,
        box: TrainingBoxSpec,
    ) -> Tuple[float, float, float]:
        if index == self.active_box_index:
            return (
                self.support_xy[0],
                self.support_xy[1],
                self.support_top_z + box.half_extents_m[2],
            )
        x, y = _inactive_box_xy(index, self.support_xy)
        return x, y, box.half_extents_m[2]

    def _position_boxes(self) -> None:
        for index, box in enumerate(self.box_specs):
            qpos_addr, qvel_addr = self._box_joint_state[box.body_name]
            x, y, z = self._box_initial_position(index, box)
            self.sim.data.qpos[qpos_addr : qpos_addr + 7] = [x, y, z, 1.0, 0.0, 0.0, 0.0]
            self.sim.data.qvel[qvel_addr : qvel_addr + 6] = 0.0
        self.sim.forward()


class RandomizedBoxSimController(SimController):
    def __init__(
        self,
        pid_controllers: Sequence[PIDController],
        model_path: str,
        box_template: TrainingBoxSpec,
        support_xy: Tuple[float, float],
        support_top_z: float,
        render: bool = True,
    ) -> None:
        super().__init__(pid_controllers=pid_controllers, model_path=model_path, render=render)
        self.box_template = box_template
        self.support_xy = support_xy
        self.support_top_z = support_top_z
        self.body_id = self.sim.model.body_name2id(box_template.body_name)
        self.geom_id = self.sim.model.geom_name2id(box_template.geom_name)
        self.qpos_addr, self.qvel_addr = self._free_joint_state_for_body(box_template.body_name)
        self.active_box = BoxSample(
            half_extents_m=box_template.half_extents_m,
            mass_kg=box_template.mass_kg,
            rgba=box_template.rgba,
        )
        self._apply_box_sample(self.active_box)
        self._passive_joint_state = self._capture_passive_joint_state()

    def set_box_sample(self, box_sample: BoxSample) -> None:
        self.active_box = box_sample
        self._apply_box_sample(box_sample)
        self._passive_joint_state = self._capture_passive_joint_state()

    def set_initial_position(self, initial_pos: list[float]) -> None:
        super().set_initial_position(initial_pos)
        self._apply_box_sample(self.active_box)
        self._passive_joint_state = self._capture_passive_joint_state()

    def _free_joint_state_for_body(self, body_name: str) -> Tuple[int, int]:
        body_id = self.sim.model.body_name2id(body_name)
        joint_id = int(self.sim.model.body_jntadr[body_id])
        if joint_id < 0:
            raise ValueError(f"{body_name} must have a free joint")
        qpos_len, qvel_len = self._joint_state_lengths(joint_id)
        if qpos_len != 7 or qvel_len != 6:
            raise ValueError(f"{body_name} must have a free joint")
        return (
            int(self.sim.model.jnt_qposadr[joint_id]),
            int(self.sim.model.jnt_dofadr[joint_id]),
        )

    def _apply_box_sample(self, box_sample: BoxSample) -> None:
        model = self.sim.model
        half_extents = np.asarray(box_sample.half_extents_m, dtype=np.float64)

        model.geom_size[self.geom_id, :3] = half_extents
        model.geom_rgba[self.geom_id, :4] = np.asarray(box_sample.rgba, dtype=np.float64)
        if hasattr(model, "geom_rbound"):
            model.geom_rbound[self.geom_id] = float(np.linalg.norm(half_extents))

        model.body_mass[self.body_id] = box_sample.mass_kg
        model.body_inertia[self.body_id, :3] = np.asarray(
            _box_inertia_diagonal(box_sample.half_extents_m, box_sample.mass_kg),
            dtype=np.float64,
        )
        if hasattr(model, "body_subtreemass"):
            model.body_subtreemass[self.body_id] = box_sample.mass_kg

        self.sim.data.qpos[self.qpos_addr : self.qpos_addr + 7] = [
            self.support_xy[0],
            self.support_xy[1],
            self.support_top_z + box_sample.half_extents_m[2],
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        self.sim.data.qvel[self.qvel_addr : self.qvel_addr + 6] = 0.0
        self.sim.forward()


class RandomBoxGraspEnv:
    def __init__(
        self,
        env: GraspPPOEnv,
        controller: MultiBoxSimController,
        box_specs: Sequence[TrainingBoxSpec],
        seed: Optional[int] = None,
    ) -> None:
        self.env = env
        self.controller = controller
        self.box_specs = tuple(box_specs)
        self.rng = np.random.default_rng(seed)
        self.active_box = self.box_specs[0]

    @property
    def observation_dim(self) -> int:
        return self.env.observation_dim

    @property
    def num_actions(self) -> int:
        return self.env.num_actions

    @property
    def steps(self) -> int:
        return self.env.steps

    @property
    def last_info(self):
        return self.env.last_info

    def reset(self):
        box_index = int(self.rng.integers(len(self.box_specs)))
        self.active_box = self.box_specs[box_index]
        self.controller.set_active_box(box_index)
        self.env.env_config.default_body_name = self.active_box.body_name
        return self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        info = {
            **info,
            "box_name": self.active_box.name,
            "box_mass_kg": self.active_box.mass_kg,
            "box_half_extents_m": self.active_box.half_extents_m,
        }
        self.env.last_info = info
        return observation, reward, done, info

    def observe(self):
        return self.env.observe()

    def observation_dict(self):
        return self.env.observation_dict()


class RandomizedBoxGraspEnv:
    def __init__(
        self,
        env: GraspPPOEnv,
        controller: RandomizedBoxSimController,
        box_distribution: BoxDistribution,
        seed: Optional[int] = None,
    ) -> None:
        self.env = env
        self.controller = controller
        self.box_distribution = box_distribution
        self.rng = np.random.default_rng(seed)
        self.active_box = controller.active_box
        self.env.env_config.default_body_name = controller.box_template.body_name

    @property
    def observation_dim(self) -> int:
        return self.env.observation_dim

    @property
    def num_actions(self) -> int:
        return self.env.num_actions

    @property
    def steps(self) -> int:
        return self.env.steps

    @property
    def last_info(self):
        return self.env.last_info

    def reset(self):
        self.active_box = self.box_distribution.sample(self.rng)
        self.controller.set_box_sample(self.active_box)
        self.env.env_config.default_body_name = self.controller.box_template.body_name
        return self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        info = {
            **info,
            "box_mass_kg": self.active_box.mass_kg,
            "box_half_extents_m": self.active_box.half_extents_m,
            "box_rgba": self.active_box.rgba,
        }
        self.env.last_info = info
        return observation, reward, done, info

    def observe(self):
        return self.env.observe()

    def observation_dict(self):
        return self.env.observation_dict()


@dataclass(frozen=True)
class TrainingEnvFactory:
    mode: str
    model_path: str
    env_config: GraspEnvConfig
    observation_config: ObservationConfig
    reward_config: RewardConfig
    seed: Optional[int]
    render: bool
    support_xy: Optional[Tuple[float, float]] = None
    support_top_z: Optional[float] = None
    box_specs: Tuple[TrainingBoxSpec, ...] = ()
    box_distribution: Optional[BoxDistribution] = None

    def __call__(self):
        pid_controllers = [
            PIDController(0.01, 0.0, 0.0) for _ in range(8)
        ]
        if self.mode == "single":
            controller = SimController(
                pid_controllers=pid_controllers,
                model_path=self.model_path,
                render=self.render,
            )
            return GraspPPOEnv(
                controller=controller,
                object_height_fn=lambda controller: None,
                env_config=self.env_config,
                observation_config=self.observation_config,
                reward_config=self.reward_config,
                seed=self.seed,
            )

        if self.support_xy is None or self.support_top_z is None:
            raise ValueError(f"{self.mode} training requires support geometry")

        if self.mode == "fixed":
            controller = MultiBoxSimController(
                pid_controllers=pid_controllers,
                model_path=self.model_path,
                box_specs=self.box_specs,
                support_xy=self.support_xy,
                support_top_z=self.support_top_z,
                render=self.render,
            )
            base_env = GraspPPOEnv(
                controller=controller,
                env_config=self.env_config,
                observation_config=self.observation_config,
                reward_config=self.reward_config,
                seed=self.seed,
            )
            return RandomBoxGraspEnv(
                env=base_env,
                controller=controller,
                box_specs=self.box_specs,
                seed=None if self.seed is None else self.seed + 1,
            )

        if self.mode == "randomized":
            if self.box_distribution is None:
                raise ValueError("Randomized training requires a box distribution")
            controller = RandomizedBoxSimController(
                pid_controllers=pid_controllers,
                model_path=self.model_path,
                box_template=RANDOMIZED_BOX_TEMPLATE,
                support_xy=self.support_xy,
                support_top_z=self.support_top_z,
                render=self.render,
            )
            base_env = GraspPPOEnv(
                controller=controller,
                env_config=self.env_config,
                observation_config=self.observation_config,
                reward_config=self.reward_config,
                seed=self.seed,
            )
            return RandomizedBoxGraspEnv(
                env=base_env,
                controller=controller,
                box_distribution=self.box_distribution,
                seed=None if self.seed is None else self.seed + 1,
            )

        raise ValueError(f"Unknown training environment mode: {self.mode!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight PPO-clip gripper policy.")
    parser.add_argument("--model-path", default=str(combined_xml), help="MuJoCo XML path.")
    parser.add_argument("--save-path", default="ppo_grasp_policy.pt", help="Where to write checkpoints.")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-iters", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device (auto, cpu, cuda, or cuda:N). Defaults to PPOConfig.device.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-force", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable the MuJoCo viewer and run simulation as fast as possible.",
    )
    parser.add_argument(
        "--num-envs",
        "--num-ens",
        dest="num_envs",
        type=int,
        default=4,
        help=(
            "Number of parallel simulation environments used with --no-render "
            "(default: 4). --num-ens is accepted as an alias."
        ),
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--single-box",
        action="store_true",
        help="Train only on --model-path instead of the randomized box distribution.",
    )
    mode_group.add_argument(
        "--fixed-three-boxes",
        action="store_true",
        help="Train on the previous three fixed colored boxes instead of continuous randomization.",
    )
    parser.add_argument(
        "--box-x-half-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=DEFAULT_BOX_DISTRIBUTION.x_half_extent_range_m,
        help="Uniform range for sampled box x half-extent in meters.",
    )
    parser.add_argument(
        "--box-y-half-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=DEFAULT_BOX_DISTRIBUTION.y_half_extent_range_m,
        help="Uniform range for sampled box y half-extent in meters.",
    )
    parser.add_argument(
        "--box-z-half-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=DEFAULT_BOX_DISTRIBUTION.z_half_extent_range_m,
        help="Uniform range for sampled box z half-extent in meters.",
    )
    parser.add_argument(
        "--box-mass-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=DEFAULT_BOX_DISTRIBUTION.mass_range_kg,
        help="Uniform range for sampled box mass in kg.",
    )
    return parser.parse_args()


def _range_from_args(values: Sequence[float], name: str) -> Tuple[float, float]:
    lo, hi = float(values[0]), float(values[1])
    if lo <= 0.0 or hi <= 0.0:
        raise ValueError(f"{name} values must be positive")
    if hi < lo:
        raise ValueError(f"{name} max must be greater than or equal to min")
    return lo, hi


def main() -> None:
    args = parse_args()
    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least 1")

    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device is not None and device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested PyTorch device {device!r}, but CUDA is not available."
            )

    env_config_kwargs = {}
    if args.max_steps is not None:
        env_config_kwargs["max_steps"] = args.max_steps

    observation_config_kwargs = {}
    if args.min_force is not None:
        observation_config_kwargs["min_force_threshold_n"] = args.min_force

    mode = "single"
    training_model_path = args.model_path
    support_xy = None
    support_top_z = None
    box_specs: Tuple[TrainingBoxSpec, ...] = ()
    box_distribution = None

    if args.single_box:
        mode = "single"
    elif args.fixed_three_boxes:
        mode = "fixed"
        scene = build_three_box_scene_xml(args.model_path, TRAINING_BOXES)
        training_model_path = str(scene.path)
        support_xy = scene.support_xy
        support_top_z = scene.support_top_z
        box_specs = TRAINING_BOXES
        print(f"training_model={scene.path}")
        print(
            "training_boxes="
            + ", ".join(
                f"{box.name}(half_extents={box.half_extents_m}, mass={box.mass_kg}kg)"
                for box in TRAINING_BOXES
            )
        )
    else:
        mode = "randomized"
        box_distribution = BoxDistribution(
            x_half_extent_range_m=_range_from_args(
                args.box_x_half_range,
                "--box-x-half-range",
            ),
            y_half_extent_range_m=_range_from_args(
                args.box_y_half_range,
                "--box-y-half-range",
            ),
            z_half_extent_range_m=_range_from_args(
                args.box_z_half_range,
                "--box-z-half-range",
            ),
            mass_range_kg=_range_from_args(
                args.box_mass_range,
                "--box-mass-range",
            ),
        )
        scene = build_randomized_box_scene_xml(args.model_path, RANDOMIZED_BOX_TEMPLATE)
        randomized_env_config_kwargs = {
            **env_config_kwargs,
            "default_body_name": RANDOMIZED_BOX_TEMPLATE.body_name,
        }
        env_config_kwargs = randomized_env_config_kwargs
        training_model_path = str(scene.path)
        support_xy = scene.support_xy
        support_top_z = scene.support_top_z
        print(f"training_model={scene.path}")
        print(
            "box_distribution="
            f"x_half={box_distribution.x_half_extent_range_m}m, "
            f"y_half={box_distribution.y_half_extent_range_m}m, "
            f"z_half={box_distribution.z_half_extent_range_m}m, "
            f"mass={box_distribution.mass_range_kg}kg"
        )

    env_config = GraspEnvConfig(**env_config_kwargs)
    observation_config = ObservationConfig(**observation_config_kwargs)
    reward_config = RewardConfig()

    def env_factory(worker_index: int, render: bool) -> TrainingEnvFactory:
        worker_seed = (
            None if args.seed is None else args.seed + worker_index * 2
        )
        return TrainingEnvFactory(
            mode=mode,
            model_path=training_model_path,
            env_config=env_config,
            observation_config=observation_config,
            reward_config=reward_config,
            seed=worker_seed,
            render=render,
            support_xy=support_xy,
            support_top_z=support_top_z,
            box_specs=box_specs,
            box_distribution=box_distribution,
        )

    if args.no_render and args.num_envs > 1:
        env = SubprocessVectorEnv(
            [env_factory(index, render=False) for index in range(args.num_envs)]
        )
        trainer_type = ParallelPPOTrainer
        active_num_envs = args.num_envs
    else:
        env = env_factory(0, render=not args.no_render)()
        trainer_type = PPOTrainer
        active_num_envs = 1
        if not args.no_render and args.num_envs != 4:
            print("num_envs_ignored_without_no_render=true")

    ppo_config_kwargs = {}
    if args.learning_rate is not None:
        ppo_config_kwargs["learning_rate"] = args.learning_rate
    if args.rollout_steps is not None:
        ppo_config_kwargs["rollout_steps"] = args.rollout_steps
    if args.batch_size is not None:
        ppo_config_kwargs["batch_size"] = args.batch_size
    if args.train_iters is not None:
        ppo_config_kwargs["train_iters"] = args.train_iters
    if device is not None:
        ppo_config_kwargs["device"] = device

    config = PPOConfig(
        observation_dim=env.observation_dim,
        num_actions=env.num_actions,
        **ppo_config_kwargs,
    )
    total_timesteps = args.total_timesteps if args.total_timesteps is not None else 100000
    print(f"training_device={config.device}")
    print(f"training_num_envs={active_num_envs}")
    agent = PPOAgent(config)
    agent.environment_metadata = GraspPPOEnv.policy_metadata_for_configs(
        env_config,
        observation_config,
    )
    try:
        trainer = trainer_type(env, agent, config)
        trainer.train(total_timesteps=total_timesteps, save_path=Path(args.save_path))
    finally:
        if isinstance(env, SubprocessVectorEnv):
            env.close()


if __name__ == "__main__":
    main()
