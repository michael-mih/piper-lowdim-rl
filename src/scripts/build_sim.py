from pathlib import Path
import xml.etree.ElementTree as ET

import os

repo_root = Path(__file__).resolve().parents[2]
robot_xml = repo_root / "piper_ros/src/piper_description/mujoco_model/piper_description.xml"
box_half_extents = "0.02 0.01 0.04"
support_half_extents = "0.035 0.025 0.04"
finger_pad_position = "0 -0.02 -0.0005"
finger_pad_half_extents = "0.015 0.015 0.001"
finger_pad_friction = "1 1 0.1 0.01 0.01"


current_path = os.path.dirname(os.path.realpath(__file__))
xml_dir = os.path.join(current_path, '..', '..', 'piper_ros', 'src', 'piper_description', 'mujoco_model')

combined_xml = xml_dir + "/tmp_scene.xml"

tree = ET.parse(robot_xml)
root = tree.getroot()

# Resolve friction constraints accurately enough for sustained pinch grasps.
# Elliptic cones and no-slip iterations avoid the slow tangential creep seen
# with the default pyramidal approximation.
option = root.find("option")
if option is None:
    option = ET.SubElement(root, "option")
option.set("cone", "elliptic")
option.set("impratio", "10")
option.set("noslip_iterations", "10")
option.set("noslip_tolerance", "1e-8")


def find_body(body, name):
    if body.get("name") == name:
        return body
    for child in body.findall("body"):
        found = find_body(child, name)
        if found is not None:
            return found
    return None


worldbody = root.find("worldbody")
link7 = find_body(worldbody, "link7")
link8 = find_body(worldbody, "link8")

# The source model uses very stiff, unlimited position actuators for the
# fingers. Make the test-scene gripper compliant and cap its closing force so a
# position error cannot crush or eject the object during arm motion.
actuator = root.find("actuator")
if actuator is None:
    raise ValueError("Could not find robot actuators")
for position_actuator in actuator.findall("position"):
    if position_actuator.get("joint") in ("joint7", "joint8"):
        position_actuator.set("kp", "1000")
        position_actuator.set("forcelimited", "true")
        # The old FSM used 2 N actuators, which cannot support the current
        # 0.6 kg upper training mass even at the friction limit.
        position_actuator.set("forcerange", "-4 4")

pad_names = []
for finger_body, side in (
    (link7, "left"),
    (link8, "right"),
):
    site_name = f"{side}_gripper_touch_site"
    pad_name = f"{side}_gripper_pad"
    if finger_body is None:
        raise ValueError(f"Could not find gripper body for {site_name}")

    # Keep the detailed finger mesh for rendering, but use a centered primitive
    # pad for collision. Mesh contacts landed at different heights on each
    # finger and generated a torque that rotated the box out of the grasp.
    for geom in finger_body.findall("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")

    ET.SubElement(
        finger_body,
        "geom",
        {
            "name": pad_name,
            "type": "box",
            "pos": finger_pad_position,
            "size": finger_pad_half_extents,
            "condim": "6",
            "friction": "1 0.1 0.01",
            "rgba": "0.1 0.8 0.95 0.35",
        },
    )
    ET.SubElement(
        finger_body,
        "site",
        {
            "name": site_name,
            "type": "box",
            "pos": finger_pad_position,
            "size": "0.015 0.015 0.0015",
            "rgba": "0.1 0.8 0.95 0",
        },
    )
    pad_names.append(pad_name)

sensor = root.find("sensor")
if sensor is None:
    sensor = ET.SubElement(root, "sensor")

ET.SubElement(sensor, "touch", {"name": "left_gripper_force", "site": "left_gripper_touch_site"})
ET.SubElement(sensor, "touch", {"name": "right_gripper_force", "site": "right_gripper_touch_site"})

ET.SubElement(
    worldbody,
    "geom",
    {
        "name": "floor",
        "type": "plane",
        "pos": "0 0 0",
        "size": "1 1 0.05",
        "rgba": "0.45 0.48 0.52 1",
        "friction": "1 0.005 0.0001",
    },
)

ET.SubElement(
    worldbody,
    "geom",
    {
        "name": "grasp_box_support",
        "type": "box",
        "pos": "0.33 0 0.02",
        "size": support_half_extents,
        "rgba": "0.25 0.28 0.32 1",
        "friction": "1 0.005 0.0001",
    },
)

body = ET.SubElement(
    worldbody,
    "body",
    {
        "name": "grasp_box",
        # support top (0.06 m) + box half-height (0.04 m)
        "pos": "0.33 0 0.10",
    },
)

ET.SubElement(body, "freejoint")

ET.SubElement(
    body,
    "geom",
    {
        "name": "grasp_box_geom",
        "type": "box",
        "size": box_half_extents,
        "rgba": "0.9 0.15 0.08 1",
        "mass": "0.1",
        "condim": "6",
        "friction": "1 0.1 0.01",
    },
)

#ET.SubElement(
#    body,
#    "geom",
#    {
#        "name": "grasp_box_geom",
#        "type": "box",
#        "size": box_half_extents,
#        "rgba": "0.9 0.15 0.08 1",
#        "mass": "0.05",
#        "condim": "6",
#        "friction": "1 0.1 0.01",
#    },
#)

# Explicit pairs prevent geom-mixing rules from silently dropping rotational
# resistance. The five values are two sliding, one torsional, and two rolling
# friction coefficients.
contact = root.find("contact")
if contact is None:
    contact = ET.SubElement(root, "contact")
for pad_name in pad_names:
    ET.SubElement(
        contact,
        "pair",
        {
            "geom1": pad_name,
            "geom2": "grasp_box_geom",
            "condim": "6",
            "friction": finger_pad_friction,
        },
    )

tree.write(combined_xml)
