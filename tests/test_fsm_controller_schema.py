import unittest

from fsm.custom_min_grasp_fsm import FSMActor


class SevenCommandController:
    joint_bounds = [
        -1.0,
        1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        0.0,
        0.07,
    ]

    def __init__(self):
        self.command = [0.0, 0.0, 0.0, 0.0, -0.7, 0.0, 0.04]

    def get_joint_angle_cmd(self):
        return list(self.command)

    def send_joint_angle_cmd(self, command):
        if len(command) != 7:
            raise ValueError("Expected seven command values")
        self.command = list(command)


class FSMControllerSchemaTests(unittest.TestCase):
    def test_tighten_and_loosen_use_combined_gripper_gap(self):
        controller = SevenCommandController()
        actor = FSMActor(controller)

        actor.tighten(0.001)
        self.assertAlmostEqual(controller.command[6], 0.038)
        actor.loosen(0.001)
        self.assertAlmostEqual(controller.command[6], 0.04)
        self.assertEqual(len(controller.command), 7)

    def test_gripper_commands_are_clamped_to_controller_bounds(self):
        controller = SevenCommandController()
        actor = FSMActor(controller)

        actor.tighten(1.0)
        self.assertEqual(controller.command[6], 0.0)
        actor.loosen(1.0)
        self.assertEqual(controller.command[6], 0.07)


if __name__ == "__main__":
    unittest.main()
