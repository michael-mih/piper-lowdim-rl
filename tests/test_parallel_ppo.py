import sys
import unittest
from unittest import mock

import numpy as np

from learning import ppo
from learning.ppo import (
    PPOAgent,
    PPOConfig,
    ParallelPPOTrainer,
    SubprocessVectorEnv,
)
from scripts.train_ppo_grasp import parse_args


class TinyEnv:
    observation_dim = 2
    num_actions = 1

    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.steps = 0

    def reset(self):
        self.steps = 0
        return np.asarray([self.worker_id, self.steps], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        observation = np.asarray(
            [self.worker_id, self.steps],
            dtype=np.float32,
        )
        done = self.steps >= 3
        info = {
            "left_force_n": 0.2,
            "right_force_n": 0.2,
            "success": done,
            "truncated": False,
            "reason": "success" if done else None,
        }
        return observation, float(action[0]), done, info


class TinyEnvFactory:
    def __init__(self, worker_id):
        self.worker_id = worker_id

    def __call__(self):
        return TinyEnv(self.worker_id)


class SubprocessVectorEnvTests(unittest.TestCase):
    def test_steps_selected_workers(self):
        env = SubprocessVectorEnv([TinyEnvFactory(0), TinyEnvFactory(1)])
        try:
            observations = env.reset()
            self.assertEqual(observations.shape, (2, 2))
            np.testing.assert_array_equal(observations[:, 0], [0.0, 1.0])

            next_observations, rewards, dones, infos = env.step(
                [1],
                np.asarray([[0.5]], dtype=np.float32),
            )
            np.testing.assert_array_equal(next_observations, [[1.0, 1.0]])
            np.testing.assert_allclose(rewards, [0.5])
            np.testing.assert_array_equal(dones, [False])
            self.assertEqual(len(infos), 1)
        finally:
            env.close()

    @unittest.skipIf(ppo.torch is None, "PyTorch is not installed")
    def test_parallel_trainer_collects_exact_rollout_size(self):
        env = SubprocessVectorEnv(
            [TinyEnvFactory(0), TinyEnvFactory(1), TinyEnvFactory(2)]
        )
        try:
            config = PPOConfig(
                observation_dim=2,
                num_actions=1,
                rollout_steps=11,
                train_iters=1,
                batch_size=11,
            )
            agent = PPOAgent(config)
            trainer = ParallelPPOTrainer(env, agent, config)
            data, rollout_info = trainer.collect_rollout()

            self.assertEqual(tuple(data["obs"].shape), (11, 2))
            self.assertEqual(tuple(data["act"].shape), (11, 1))
            self.assertEqual(rollout_info["episodes"], 3.0)
        finally:
            env.close()


class ParallelTrainingArgumentTests(unittest.TestCase):
    def test_num_envs_defaults_to_four(self):
        with mock.patch.object(sys, "argv", ["train-ppo-grasp"]):
            self.assertEqual(parse_args().num_envs, 4)

    def test_num_ens_alias_is_accepted(self):
        with mock.patch.object(
            sys,
            "argv",
            ["train-ppo-grasp", "--num-ens", "7"],
        ):
            self.assertEqual(parse_args().num_envs, 7)


if __name__ == "__main__":
    unittest.main()
