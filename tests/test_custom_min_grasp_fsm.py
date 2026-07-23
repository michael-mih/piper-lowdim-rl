import unittest

from fsm.custom_min_grasp_fsm import FSMActor


class ForceController:
    def __init__(self, left_force=1.0, right_force=1.0):
        self.left_force = left_force
        self.right_force = right_force

    def get_force_left(self):
        return self.left_force

    def get_force_right(self):
        return self.right_force

    def get_force_average(self):
        return 0.5 * (self.left_force + self.right_force)

    def step(self):
        pass


class FSMActorTests(unittest.TestCase):
    def setUp(self):
        self.controller = ForceController()
        self.actor = FSMActor(self.controller)
        self.actor.reset_tracking()

    def test_cumulative_force_drop_across_window_is_slip(self):
        self.actor.three_force_buffer_left = [0.7, 0.85, 1.0]

        self.assertTrue(self.actor.is_slip())

    def test_small_force_drop_is_not_slip(self):
        self.actor.three_force_buffer_left = [0.85, 0.9, 1.0]

        self.assertFalse(self.actor.is_slip())

    def test_three_low_force_samples_are_slip(self):
        self.actor.three_force_buffer_right = [0.19, 0.18, 0.17]

        self.assertTrue(self.actor.is_slip())

    def test_reset_tracking_discards_previous_trial(self):
        self.actor.three_force_buffer_left = [0.1, 0.5, 1.0]
        self.controller.left_force = 0.8
        self.controller.right_force = 0.9

        self.actor.reset_tracking()

        self.assertEqual(self.actor.three_force_buffer_left, [0.8] * 3)
        self.assertEqual(self.actor.three_force_buffer_right, [0.9] * 3)
        self.assertFalse(self.actor.is_slip())

    def test_convergence_requires_no_new_minimum(self):
        self.assertFalse(self.actor.is_converged(0))
        self.actor.step_count = 1

        self.assertTrue(self.actor.is_converged(0))


if __name__ == "__main__":
    unittest.main()
