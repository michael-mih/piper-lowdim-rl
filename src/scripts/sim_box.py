from __future__ import annotations

import numpy as np


def set_box_mass(
    controller,
    mass_kg: float,
    body_name: str = "grasp_box",
    geom_name: str = "grasp_box_geom",
) -> None:
    mass_kg = float(mass_kg)
    if mass_kg <= 0.0:
        raise ValueError("Box mass must be greater than zero")

    model = controller.sim.model
    body_id = model.body_name2id(body_name)
    geom_id = model.geom_name2id(geom_name)
    hx, hy, hz = np.asarray(model.geom_size[geom_id, :3], dtype=np.float64)

    model.body_mass[body_id] = mass_kg
    model.body_inertia[body_id, :3] = (mass_kg / 3.0) * np.asarray(
        (
            hy * hy + hz * hz,
            hx * hx + hz * hz,
            hx * hx + hy * hy,
        ),
        dtype=np.float64,
    )
    if hasattr(model, "body_subtreemass"):
        model.body_subtreemass[body_id] = mass_kg
    controller.sim.forward()
