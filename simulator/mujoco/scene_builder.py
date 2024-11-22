import os
import random
import pathlib
import shutil
from typing import Tuple, Union
import xml.etree.ElementTree as ET

import fire
import numpy as np
import robosuite
import robosuite.models
import robosuite.models.arenas
from robosuite.models.objects import BoxObject
import robosuite.models.robots

RANGE_OR_VALUE = Union[float, Tuple[float, float]]

base_actuator_values = np.array([
    [0.3314, -0.4919, -0.5844, -2.0910, -0.2424, 1.7946, 0.6114, 0.0000],
    [-0.2563, 0.5006, -0.0556, -0.6951, 0.0673, 1.3190, 0.5028, 0.0000],
    [-0.0885, -0.4360, 0.0397, -2.3905, 0.0269, 2.0837, 0.7222, 0.0000]
])

def random_color():
    return [random.random() for _ in range(3)] + [1]


def generate_initial_actuator_values():
    """
    Generate initial actuator values as random linear combination of base actuator values.
    """
    weights = np.random.rand(base_actuator_values.shape[0])
    weights /= weights.sum()
    return (base_actuator_values * weights[:, None]).sum(axis=0).tolist()


def random_range(value: RANGE_OR_VALUE) -> float:
    if isinstance(value, float):
        return value

    try:
        return random.uniform(value[0], value[1])
    except TypeError:
        raise ValueError(f"Invalid range: {value}")


def generate_scene(
    path: str,
    num_boxes: int = 2,
    table_height: RANGE_OR_VALUE = (0.1, 0.2),
    table_size: Tuple[float, float, float] = (0.4, 0.6, 0.05),
    box_size: RANGE_OR_VALUE = 0.02,

):
    table_height = random_range(table_height)

    table_offset = (
        -0.3,  # make table closer to robot
        0,
        table_height  # make table lower
    )

    world = robosuite.models.MujocoWorldBase()

    mujoco_arena = robosuite.models.arenas.TableArena(
        table_full_size=table_size,
        table_friction=(1, 0.005, 0.0001),
        table_offset=table_offset,
    )
    mujoco_arena.set_origin([0.8, 0, 0])
    world.merge(mujoco_arena)

    box_size = random_range(box_size)
    box_pos = []

    for i in range(num_boxes):
        object = BoxObject(
            name=f"box_{i}",
            size=[box_size, box_size, box_size * 0.5],
            rgba=random_color(),
            density=2500,

        ).get_obj()
        tabletop_x, tabletop_y, tabletop_z = mujoco_arena.table_top_abs  # table top midpoint

        tabletop_z += box_size / 2  # make box sit on table
        tabletop_x += random.uniform(-table_size[0] / 2 + box_size, table_size[0] / 2 - box_size * 2)
        tabletop_y += random.uniform(-table_size[1] / 2 + box_size, table_size[1] / 2 - box_size * 2)

        object.set('pos', f'{tabletop_x} {tabletop_y} {tabletop_z}')
        world.worldbody.append(object)

        box_pos.extend([tabletop_x, tabletop_y, tabletop_z, 1, 0, 0, 0])

    with open(path, "w") as f:
        f.write(world.get_xml())

    tree = ET.parse(path)
    root = tree.getroot()

    # add robot to scene
    root.insert(0, ET.Element("include", file="mjx_panda.xml"))

    # remove <compiler> section
    compiler = root.find("compiler")
    root.remove(compiler)

    # add robot home keyframe
    home_keyframe = ET.Element("keyframe")
    keyframe_actuator = generate_initial_actuator_values()

    qpos_str = " ".join(map(str, keyframe_actuator + [0.04] + box_pos))
    ctrl_str = " ".join(map(str, keyframe_actuator))
    home_keyframe.append(ET.Element("key", name="home", qpos=qpos_str, ctrl=ctrl_str))
    root.append(home_keyframe)

    # add to <visual>
    visual = root.find("visual")
    visual.append(ET.Element("global", azimuth="120", elevation="-20", offwidth="1920", offheight="1080"))

    # replace light
    light = root.find("worldbody/light")
    light.set("pos", "1.8 3.0 1.5")
    light.set("dir", "-0.2 -0.2 0")
    light.set("specular", "0.3 0.3 0.3")
    light.set("directional", "false")
    light.set("castshadow", "false")

    tree.write(path)

    return world

def copy_assets(scene_dir: pathlib.Path, source_dir: pathlib.Path = pathlib.Path("assets/mujoco")):
    """
    Copy assets from source directory to scene directory.

    Args:
        scene_dir: Directory to copy assets to.
        source_dir: Directory to copy assets from.
    """
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "assets").mkdir(parents=True, exist_ok=True)

    # copy assets
    for file in os.listdir(source_dir / "assets"):
        shutil.copy(source_dir / "assets" / file, scene_dir / "assets")

    shutil.copy(source_dir / "mjx_panda.xml", scene_dir)


def construct_scene(scene_path: Union[str, pathlib.Path]):
    scene_path = pathlib.Path(scene_path)
    copy_assets(scene_path.parent)
    generate_scene(scene_path)

if __name__ == "__main__":
    fire.Fire(construct_scene)
