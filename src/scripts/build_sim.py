from pathlib import Path
import xml.etree.ElementTree as ET

import os

repo_root = Path(__file__).resolve().parents[2]
robot_xml = repo_root / "piper_ros/src/piper_description/mujoco_model/piper_description.xml"
box_half_extents = "0.02 0.01 0.08"
support_half_extents = "0.035 0.025 0.02"


current_path = os.path.dirname(os.path.realpath(__file__))
xml_dir = os.path.join(current_path, '..', '..', 'piper_ros', 'src', 'piper_description', 'mujoco_model')

combined_xml = xml_dir + "/tmp_scene.xml"

tree = ET.parse(robot_xml)
root = tree.getroot()


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

for finger_body, site_name in (
    (link7, "left_gripper_touch_site"),
    (link8, "right_gripper_touch_site"),
):
    if finger_body is None:
        raise ValueError(f"Could not find gripper body for {site_name}")
    ET.SubElement(
        finger_body,
        "site",
        {
            "name": site_name,
            "type": "box",
            "pos": "0.0 -0.000 0",
            "size": "0.02 0.014 0.004",
            "rgba": "0.1 0.8 0.95 0",
        },
    )

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
        "pos": "0.35 0 0.02",
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
        "pos": "0.35 0 0.12",
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
        "mass": "0.05",
        "friction": "1 0.005 0.0001",
    },
)

tree.write(combined_xml)
