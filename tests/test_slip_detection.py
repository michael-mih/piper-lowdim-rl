import unittest

from learning.env import GraspPPOEnv, ObservationConfig, RewardConfig


class ForceController:
    ground_truth_pos = False

    def __init__(self, left_force=1.0, right_force=1.0):
        self.left_force = left_force
        self.right_force = right_force
        self.joint5 = -0.8

    def get_force_left(self):
        return self.left_force

    def get_force_right(self):
        return self.right_force

    def get_force_average(self):
        return 0.5 * (self.left_force + self.right_force)

    def get_joint_angles(self):
        return [0.0, 0.0, 0.0, 0.0, self.joint5, 0.0, 0.04]


class SlipDetectionTests(unittest.TestCase):
    def setUp(self):
        self.controller = ForceController()
        self.env = GraspPPOEnv(
            controller=self.controller,
            observation_config=ObservationConfig(min_force_threshold_n=0.05),
        )
        self.env._previous_left_force = self.controller.left_force
        self.env._previous_right_force = self.controller.right_force

    def sample(self, left_force, right_force):
        self.controller.left_force = left_force
        self.controller.right_force = right_force
        slipping = self.env._is_slip()
        self.env._previous_left_force = left_force
        self.env._previous_right_force = right_force
        return slipping

    def test_large_left_force_drop_is_slipping(self):
        self.assertTrue(self.sample(0.70, 1.0))

    def test_large_right_force_drop_is_slipping(self):
        self.assertTrue(self.sample(1.0, 0.70))

    def test_stable_or_increasing_force_is_not_slipping(self):
        self.assertFalse(self.sample(1.0, 1.0))
        self.assertFalse(self.sample(1.02, 1.03))

    def test_force_drop_is_aggregated_over_three_deltas(self):
        self.assertFalse(self.sample(0.90, 1.0))
        self.assertFalse(self.sample(0.80, 1.0))
        self.assertTrue(self.sample(0.69, 1.0))

    def test_upward_movement_during_slip_incurs_penalty(self):
        self.env._lifting_this_step = True
        self.env._upward_action_magnitude = 1.0
        self.controller.left_force = 0.70

        reward = self.env._compute_height_reward(RewardConfig())

        self.assertEqual(reward, -0.001)

    def test_slip_penalty_scales_with_upward_action(self):
        self.env._lifting_this_step = True
        self.env._upward_action_magnitude = 0.1
        self.controller.left_force = 0.70

        reward = self.env._compute_height_reward(RewardConfig())

        self.assertAlmostEqual(reward, -0.0001)

    def test_non_upward_movement_during_slip_is_not_penalized(self):
        self.env._lifting_this_step = False
        self.controller.left_force = 0.70

        reward = self.env._compute_height_reward(RewardConfig())

        self.assertEqual(reward, 0.0)

    def test_force_decrease_without_slip_is_rewarded(self):
        config = RewardConfig()

        reward = self.env._compute_force_decrease_reward(
            config,
            left_force=0.9,
            right_force=0.9,
            is_slipping=False,
        )

        self.assertAlmostEqual(reward, 0.1 * config.force_change_reward_coef)

    def test_force_decrease_during_slip_is_not_rewarded(self):
        reward = self.env._compute_force_decrease_reward(
            RewardConfig(),
            left_force=0.7,
            right_force=1.0,
            is_slipping=True,
        )

        self.assertEqual(reward, 0.0)

    def test_force_decrease_reward_is_capped(self):
        config = RewardConfig(max_rewarded_force_change_n=0.05)

        reward = self.env._compute_force_decrease_reward(
            config,
            left_force=0.5,
            right_force=0.5,
            is_slipping=False,
        )

        self.assertAlmostEqual(reward, 0.05 * config.force_change_reward_coef)

    def test_increasing_bilateral_force_recovers_after_confirmation(self):
        config = RewardConfig()
        self.env._previous_desired_force_n = 1.0
        self.assertTrue(self.sample(0.70, 1.0))
        self.assertTrue(self.sample(0.70, 1.0))
        self.assertTrue(self.sample(0.70, 1.0))

        # Positive deltas larger than the old 0.05 N stability threshold are
        # evidence of a tightening regrasp and must not block recovery.
        self.assertTrue(self.sample(0.90, 1.0))
        self.assertFalse(self.sample(1.00, 1.0))

        self.assertFalse(self.env._is_slipping_state)
        self.assertTrue(self.env._slip_recovered_this_step)
        self.assertFalse(self.env._slip_detection_armed)

    def test_slip_detection_rearms_after_three_stable_steps(self):
        self.env._previous_desired_force_n = 1.0
        self.assertTrue(self.sample(0.70, 1.0))
        self.assertTrue(self.sample(0.70, 1.0))
        self.assertTrue(self.sample(0.70, 1.0))
        self.assertTrue(self.sample(0.90, 1.0))
        self.assertFalse(self.sample(1.00, 1.0))

        # A new drop during the cooldown does not immediately relatch slip.
        self.assertFalse(self.sample(0.70, 1.0))
        self.assertFalse(self.env._slip_detection_armed)

        self.env._left_force_delta_buffer.clear()
        self.env._left_force_delta_buffer.extend((0.0, 0.0, 0.0))
        self.env._right_force_delta_buffer.clear()
        self.env._right_force_delta_buffer.extend((0.0, 0.0, 0.0))
        self.assertFalse(self.sample(0.70, 1.0))
        self.assertFalse(self.sample(0.70, 1.0))
        self.assertFalse(self.sample(0.70, 1.0))
        self.assertTrue(self.env._slip_detection_armed)
        self.assertTrue(self.sample(0.40, 1.0))

    def test_recovery_bonus_requires_new_maximum_lift_progress(self):
        config = RewardConfig()
        self.env._initial_joint5_position = -0.7
        self.env._slip_recovered_this_step = True

        self.assertEqual(
            self.env._compute_slip_recovery_bonus(config),
            config.slip_recovery_bonus,
        )

        # Cycling at the same height cannot collect another recovery bonus.
        self.env._slip_recovered_this_step = True
        self.assertEqual(self.env._compute_slip_recovery_bonus(config), 0.0)

        # A new episode-best height unlocks the next recovery bonus.
        self.controller.joint5 = -0.83
        self.env._slip_recovered_this_step = True
        self.assertEqual(
            self.env._compute_slip_recovery_bonus(config),
            config.slip_recovery_bonus,
        )

    def test_slip_state_is_an_observation(self):
        self.assertIn("is_slipping", self.env.observation_names)


if __name__ == "__main__":
    unittest.main()
