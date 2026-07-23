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

    def test_bilateral_force_collapse_is_slipping(self):
        self.assertTrue(self.sample(0.39, 0.39))

    def test_unilateral_force_collapse_is_not_slipping(self):
        self.assertFalse(self.sample(0.39, 1.0))

    def test_stable_or_increasing_force_is_not_slipping(self):
        self.assertFalse(self.sample(1.0, 1.0))
        self.assertFalse(self.sample(1.02, 1.03))

    def test_normal_force_oscillation_is_not_slipping(self):
        self.assertFalse(self.sample(0.60, 1.0))
        self.assertFalse(self.sample(1.00, 1.0))
        self.assertFalse(self.sample(0.60, 1.0))

    def test_large_drop_without_force_collapse_is_not_slipping(self):
        self.env._previous_left_force = 4.0

        self.assertFalse(self.sample(3.2, 1.0))

    def test_gradual_force_drop_is_not_slipping(self):
        self.assertFalse(self.sample(0.70, 1.0))
        self.assertFalse(self.sample(0.40, 1.0))
        self.assertFalse(self.sample(0.10, 1.0))

    def test_contact_loss_is_slipping(self):
        self.assertTrue(self.sample(0.009, 0.009))

    def test_unilateral_contact_loss_is_not_slipping(self):
        self.assertFalse(self.sample(0.009, 1.0))

    def test_upward_movement_during_slip_incurs_penalty(self):
        self.env._lifting_this_step = True
        self.env._upward_action_magnitude = 1.0
        self.controller.left_force = 0.39
        self.controller.right_force = 0.39

        reward = self.env._compute_height_reward(RewardConfig())

        self.assertEqual(reward, -0.001)

    def test_slip_penalty_scales_with_upward_action(self):
        self.env._lifting_this_step = True
        self.env._upward_action_magnitude = 0.1
        self.controller.left_force = 0.39
        self.controller.right_force = 0.39

        reward = self.env._compute_height_reward(RewardConfig())

        self.assertAlmostEqual(reward, -0.0001)

    def test_non_upward_movement_during_slip_is_not_penalized(self):
        self.env._lifting_this_step = False
        self.controller.left_force = 0.39
        self.controller.right_force = 0.39

        reward = self.env._compute_height_reward(RewardConfig())

        self.assertEqual(reward, 0.0)

    def update_force_decrease_confirmation(self, desired_force, is_slipping=False):
        return self.env._update_force_decrease_confirmation(
            RewardConfig(),
            left_force=self.controller.left_force,
            right_force=self.controller.right_force,
            is_slipping=is_slipping,
            desired_force=desired_force,
        )

    def test_desired_force_decrease_is_recorded_after_confirmation(self):
        config = RewardConfig()
        self.env._confirmed_desired_force_n = 1.0
        self.env._lifting_this_step = True

        reward = 0.0
        for _ in range(config.force_decrease_confirmation_steps - 1):
            reward += self.update_force_decrease_confirmation(0.6)
        self.assertAlmostEqual(self.env._confirmed_desired_force_n, 1.0)
        reward += self.update_force_decrease_confirmation(0.6)

        self.assertAlmostEqual(self.env._confirmed_desired_force_n, 0.6)
        self.assertGreater(reward, 0.0)

    def test_efficiency_reward_is_normalized_to_configured_maximum(self):
        config = RewardConfig()
        self.env._confirmed_desired_force_n = (
            self.env.env_config.desired_force_min_n
        )

        reward = self.env._compute_force_efficiency_reward(config)

        self.assertAlmostEqual(reward, config.max_force_efficiency_reward)

    def test_incremental_efficiency_rewards_telescope_to_total(self):
        config = RewardConfig()
        self.env._lifting_this_step = True
        earned_reward = 0.0

        for desired_force in (3.0, 2.0, 0.5):
            for _ in range(config.force_decrease_confirmation_steps):
                earned_reward += self.update_force_decrease_confirmation(
                    desired_force
                )

        self.assertAlmostEqual(
            earned_reward,
            self.env._compute_force_efficiency_reward(config),
        )
        self.assertLessEqual(earned_reward, config.max_force_efficiency_reward)

    def test_slip_cancels_pending_force_decrease(self):
        self.env._confirmed_desired_force_n = (
            self.env.env_config.desired_force_max_n
        )
        self.env._lifting_this_step = True
        self.update_force_decrease_confirmation(3.0)

        reward = self.update_force_decrease_confirmation(3.0, is_slipping=True)

        self.assertIsNone(self.env._pending_desired_force_n)
        self.assertEqual(self.env._pending_force_decrease_steps, 0)
        self.assertEqual(reward, 0.0)

    def test_force_decrease_requires_bilateral_contact(self):
        self.env._confirmed_desired_force_n = 1.0
        self.env._lifting_this_step = True
        self.controller.left_force = 0.0

        self.update_force_decrease_confirmation(0.6)
        self.assertIsNone(self.env._pending_desired_force_n)

    def test_non_lifting_steps_pause_force_decrease_confirmation(self):
        config = RewardConfig()
        self.env._confirmed_desired_force_n = 1.0
        self.env._lifting_this_step = True
        self.update_force_decrease_confirmation(0.6)

        self.env._lifting_this_step = False
        for _ in range(3):
            self.update_force_decrease_confirmation(0.6)
        self.assertEqual(self.env._pending_force_decrease_steps, 1)

        self.env._lifting_this_step = True
        reward = 0.0
        for _ in range(config.force_decrease_confirmation_steps - 1):
            reward += self.update_force_decrease_confirmation(0.6)
        self.assertGreater(reward, 0.0)

    def test_force_increase_beyond_tolerance_cancels_confirmation(self):
        self.env._confirmed_desired_force_n = 1.0
        self.env._lifting_this_step = True
        self.update_force_decrease_confirmation(0.6)

        self.update_force_decrease_confirmation(0.95)

        self.assertIsNone(self.env._pending_desired_force_n)
        self.assertEqual(self.env._pending_force_decrease_steps, 0)

    def test_small_force_jitter_can_confirm_at_highest_surviving_command(self):
        config = RewardConfig()
        self.env._confirmed_desired_force_n = 1.0
        self.env._lifting_this_step = True
        self.update_force_decrease_confirmation(0.6)

        reward = 0.0
        for _ in range(config.force_decrease_confirmation_steps - 1):
            reward += self.update_force_decrease_confirmation(0.64)

        self.assertGreater(reward, 0.0)
        self.assertAlmostEqual(self.env._confirmed_desired_force_n, 0.64)

    def test_lost_lift_progress_cancels_pending_force_decrease(self):
        self.env._confirmed_desired_force_n = 1.0
        self.env._lifting_this_step = True
        self.update_force_decrease_confirmation(0.6)

        self.controller.joint5 = -0.75
        self.update_force_decrease_confirmation(0.6)

        self.assertIsNone(self.env._pending_desired_force_n)
        self.assertEqual(self.env._pending_force_decrease_steps, 0)

    def test_slip_claws_back_confirmed_efficiency_reward(self):
        config = RewardConfig()
        self.env._confirmed_desired_force_n = (
            self.env.env_config.desired_force_max_n
        )
        self.env._lifting_this_step = True
        earned_reward = 0.0
        for _ in range(config.force_decrease_confirmation_steps):
            earned_reward += self.update_force_decrease_confirmation(0.6)
        self.assertGreater(earned_reward, 0.0)

        clawback = self.update_force_decrease_confirmation(
            0.6,
            is_slipping=True,
        )

        self.assertAlmostEqual(clawback, -earned_reward)
        self.assertAlmostEqual(
            self.env._confirmed_desired_force_n,
            self.env.env_config.desired_force_max_n,
        )
        self.assertEqual(self.env._compute_force_efficiency_reward(config), 0.0)

    def test_force_increase_claws_back_confirmed_efficiency_reward(self):
        config = RewardConfig()
        self.env._confirmed_desired_force_n = (
            self.env.env_config.desired_force_max_n
        )
        self.env._lifting_this_step = True
        earned_reward = 0.0
        for _ in range(config.force_decrease_confirmation_steps):
            earned_reward += self.update_force_decrease_confirmation(0.6)

        clawback = self.update_force_decrease_confirmation(1.0)

        self.assertLess(clawback, 0.0)
        self.assertAlmostEqual(self.env._confirmed_desired_force_n, 1.0)
        self.assertAlmostEqual(
            earned_reward + clawback,
            self.env._compute_force_efficiency_reward(config),
        )

    def test_force_reduction_cycle_cannot_farm_reward(self):
        config = RewardConfig()
        self.env._confirmed_desired_force_n = (
            self.env.env_config.desired_force_max_n
        )
        self.env._lifting_this_step = True
        net_reward = 0.0

        for _ in range(config.force_decrease_confirmation_steps):
            net_reward += self.update_force_decrease_confirmation(0.6)
        net_reward += self.update_force_decrease_confirmation(1.0)
        for _ in range(config.force_decrease_confirmation_steps):
            net_reward += self.update_force_decrease_confirmation(0.6)

        self.assertAlmostEqual(
            net_reward,
            self.env._compute_force_efficiency_reward(config),
        )

    def test_increasing_bilateral_force_recovers_after_confirmation(self):
        config = RewardConfig()
        self.env._previous_desired_force_n = 1.0
        self.assertTrue(self.sample(0.39, 0.39))
        self.assertTrue(self.sample(0.39, 0.39))
        self.assertTrue(self.sample(0.39, 0.39))

        # Positive deltas larger than the old 0.05 N stability threshold are
        # evidence of a tightening regrasp and must not block recovery.
        self.assertTrue(self.sample(0.90, 1.0))
        self.assertFalse(self.sample(1.00, 1.0))

        self.assertFalse(self.env._is_slipping_state)
        self.assertTrue(self.env._slip_recovered_this_step)
        self.assertFalse(self.env._slip_detection_armed)

    def test_new_force_collapse_relatches_during_rearm_cooldown(self):
        self.env._previous_desired_force_n = 1.0
        self.assertTrue(self.sample(0.39, 0.39))
        self.assertTrue(self.sample(0.39, 0.39))
        self.assertTrue(self.sample(0.39, 0.39))
        self.assertTrue(self.sample(0.90, 1.0))
        self.assertFalse(self.sample(1.00, 1.0))

        self.assertFalse(self.env._slip_detection_armed)
        self.assertTrue(self.sample(0.39, 0.39))
        self.assertTrue(self.env._is_slipping_state)

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
