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


class ForceChangingController(ForceController):
    def step(self):
        self.left_force = 0.3


class FSMActorTests(unittest.TestCase):
    def setUp(self):
        self.controller = ForceController()
        self.actor = FSMActor(self.controller)
        self.actor.reset_tracking()

    def test_abrupt_force_collapse_is_slip(self):
        self.actor.three_force_buffer_left = [0.39, 1.0, 1.0]
        self.actor.three_force_buffer_right = [0.39, 1.0, 1.0]

        self.assertTrue(self.actor.is_slip())

    def test_unilateral_force_collapse_is_not_slip(self):
        self.actor.three_force_buffer_left = [0.39, 1.0, 1.0]

        self.assertFalse(self.actor.is_slip())

    def test_gradual_force_drop_is_not_slip(self):
        self.actor.three_force_buffer_left = [0.4, 0.7, 1.0]

        self.assertFalse(self.actor.is_slip())

    def test_normal_force_oscillation_is_not_slip(self):
        self.actor.three_force_buffer_left = [1.6, 2.0, 1.8]

        self.assertFalse(self.actor.is_slip())

    def test_large_drop_without_force_collapse_is_not_slip(self):
        self.actor.three_force_buffer_left = [3.2, 4.0, 4.0]

        self.assertFalse(self.actor.is_slip())

    def test_low_but_nonzero_contact_is_not_slip(self):
        self.actor.three_force_buffer_right = [0.19, 0.18, 0.17]
        self.controller.right_force = 0.17

        self.assertFalse(self.actor.is_slip())

    def test_contact_loss_is_slip(self):
        self.controller.left_force = 0.009
        self.controller.right_force = 0.009

        self.assertTrue(self.actor.is_slip())

    def test_unilateral_contact_loss_is_not_slip(self):
        self.controller.right_force = 0.009

        self.assertFalse(self.actor.is_slip())

    def test_step_samples_force_before_advancing_controller(self):
        controller = ForceChangingController()
        actor = FSMActor(controller)
        actor.reset_tracking()

        actor.step()

        self.assertEqual(actor.three_force_buffer_left[0], 1.0)
        self.assertEqual(controller.left_force, 0.3)

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
