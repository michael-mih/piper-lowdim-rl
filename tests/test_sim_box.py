import unittest

import numpy as np

from scripts.sim_box import set_box_mass


class FakeModel:
    def __init__(self):
        self.geom_size = np.asarray([[0.02, 0.01, 0.04]], dtype=np.float64)
        self.body_mass = np.asarray([0.1], dtype=np.float64)
        self.body_inertia = np.zeros((1, 3), dtype=np.float64)
        self.body_subtreemass = np.asarray([0.1], dtype=np.float64)

    def body_name2id(self, name):
        if name != "grasp_box":
            raise KeyError(name)
        return 0

    def geom_name2id(self, name):
        if name != "grasp_box_geom":
            raise KeyError(name)
        return 0


class FakeSim:
    def __init__(self):
        self.model = FakeModel()
        self.forward_calls = 0

    def forward(self):
        self.forward_calls += 1


class FakeController:
    def __init__(self):
        self.sim = FakeSim()


class SimBoxTests(unittest.TestCase):
    def test_set_box_mass_updates_mass_and_inertia(self):
        controller = FakeController()

        set_box_mass(controller, 0.6)

        model = controller.sim.model
        self.assertEqual(model.body_mass[0], 0.6)
        self.assertEqual(model.body_subtreemass[0], 0.6)
        np.testing.assert_allclose(
            model.body_inertia[0],
            (0.6 / 3.0)
            * np.asarray(
                [0.01**2 + 0.04**2, 0.02**2 + 0.04**2, 0.02**2 + 0.01**2]
            ),
        )
        self.assertEqual(controller.sim.forward_calls, 1)

    def test_box_mass_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            set_box_mass(FakeController(), 0.0)


if __name__ == "__main__":
    unittest.main()
