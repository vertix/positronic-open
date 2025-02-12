import os
import pickle
import random
import pathlib
from typing import Any, Dict, Optional, Tuple, Union

import fire
import numpy as np
import mujoco

import robosuite
import robosuite.models
import robosuite.models.arenas
from robosuite.models.objects import BoxObject

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


def extract_assets(spec: mujoco.MjSpec, parent_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract assets from the spec and return a dictionary of asset names to asset data.

    This function is used to create portable mujoco scenes, since original xml files containes
    paths to assets (sometimes relative to meshdir/texturedir and sometimes absolute).

    args:
        spec: mujoco.MjSpec to extract assets from
        parent_dir: directory to prepend to asset paths
    """

    def _load_asset_binary(asset, parent_dir: str) -> bytes:
        asset_path = os.path.join(parent_dir, asset.file)
        with open(asset_path, 'rb') as f:
            data = f.read()
        return data

    assets = {}
    for texture in spec.textures:
        if not texture.file:
            continue

        if not texture.name:
            texture.name = os.path.splitext(os.path.basename(texture.file))[0]

        texture_file_ext = os.path.splitext(texture.file)[1]
        texture_name = f"{texture.name}{texture_file_ext}"
        path = os.path.join(parent_dir, spec.texturedir) if parent_dir else spec.texturedir

        assets[texture_name] = _load_asset_binary(texture, path)
        texture.file = texture_name

    for mesh in spec.meshes:
        if not mesh.file:
            continue
        if not mesh.name:
            mesh.name = os.path.splitext(os.path.basename(mesh.file))[0]

        mesh_file_ext = os.path.splitext(mesh.file)[1]
        mesh_name = f"{mesh.name}{mesh_file_ext}"
        path = os.path.join(parent_dir, spec.meshdir) if parent_dir else spec.meshdir

        assets[mesh_name] = _load_asset_binary(mesh, path)
        mesh.file = mesh_name

    return assets


# TODO: In theory this could be implemented via series of scene transformations. But this is a bit of a pain. :_)
def generate_scene(
    num_boxes: int = 2,
    table_height: RANGE_OR_VALUE = (0.1, 0.2),
    table_size: Tuple[float, float, float] = (0.4, 0.6, 0.05),
    box_size: RANGE_OR_VALUE = 0.02,
    portable: bool = True,
) -> mujoco.MjModel:
    table_height = random_range(table_height)
    table_offset = (-0.3, 0, table_height)

    # Create base world with table
    world = robosuite.models.MujocoWorldBase()
    mujoco_arena = robosuite.models.arenas.TableArena(
        table_full_size=table_size,
        table_friction=(1, 0.005, 0.0001),
        table_offset=table_offset,
    )
    mujoco_arena.set_origin([0.8, 0, 0])
    world.merge(mujoco_arena)

    # Add boxes
    box_size = random_range(box_size)
    box_pos = []

    for i in range(num_boxes):
        object = BoxObject(
            name=f"box_{i}",
            size=[box_size, box_size, box_size * 0.5],
            rgba=random_color(),
            density=2500,
        ).get_obj()
        tabletop_x, tabletop_y, tabletop_z = mujoco_arena.table_top_abs

        tabletop_z += box_size / 2
        tabletop_x += random.uniform(-table_size[0] / 2 + box_size, table_size[0] / 2 - box_size * 2)
        tabletop_y += random.uniform(-table_size[1] / 2 + box_size, table_size[1] / 2 - box_size * 2)

        object.set('pos', f'{tabletop_x} {tabletop_y} {tabletop_z}')
        world.worldbody.append(object)
        box_pos.extend([tabletop_x, tabletop_y, tabletop_z, 1, 0, 0, 0])

    asset_dict = {}
    # Load Panda robot specification as base
    panda_spec = mujoco.MjSpec.from_file("assets/mujoco/mjx_panda.xml")
    if portable:
        panda_assets = extract_assets(panda_spec, "assets/mujoco")
        asset_dict = {**panda_assets}

    # Create temporary spec from the world to merge into Panda spec
    world_spec = mujoco.MjSpec.from_string(world.get_xml())
    if portable:
        world_assets = extract_assets(world_spec, "")
        asset_dict = {**asset_dict, **world_assets}

    # Adjust lighting
    world_spec.lights[0].pos = [1.8, 3.0, 1.5]
    world_spec.lights[0].dir = [-0.2, -0.2, 0]
    world_spec.lights[0].specular = [0.3, 0.3, 0.3]
    world_spec.lights[0].directional = 0
    world_spec.lights[0].castshadow = 0

    # add panda to world
    origin_site = world_spec.worldbody.add_site(name="origin", pos=[0, 0, 0])
    origin_site.attach_body(panda_spec.worldbody, '', '_ph')  # suffix is required by .attach()

    # Add keyframe data
    keyframe_actuator = generate_initial_actuator_values()
    qpos = box_pos + keyframe_actuator + [0.04]
    world_spec.add_key(name="home", qpos=qpos, ctrl=keyframe_actuator)

    # Configure visual settings
    g = getattr(panda_spec.visual, "global")
    g.azimuth = 120
    g.elevation = -20
    g.offwidth = 1920
    g.offheight = 1080

    if portable:
        world_spec.meshdir = "!!! This spec should be loaded with asset dict. See simulator.mujoco.transforms.load_model_from_spec !!!"
        world_spec.texturedir = "!!! This spec should be loaded with asset dict. See simulator.mujoco.transforms.load_model_from_spec !!!"

        world_spec.assets = asset_dict
    else:
        world_spec.meshdir = "assets/mujoco/assets/"

    # Using custom texts to store metadata about the model
    world_spec.add_text(name='model_suffix', data='_ph')

    return world_spec, asset_dict


def construct_scene(scene_path: Union[str, pathlib.Path]):
    scene_path = pathlib.Path(scene_path)
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    assets_path = scene_path.with_suffix('.assets.pkl')
    relative_assets_path = assets_path.name
    spec, assets = generate_scene()

    spec.add_text(name='relative_assets_path', data=str(relative_assets_path))

    spec.compile()

    with open(scene_path, "w") as f:
        f.write(spec.to_xml())

    with open(assets_path, "wb") as f:
        pickle.dump(assets, f)


if __name__ == "__main__":
    fire.Fire(construct_scene)
