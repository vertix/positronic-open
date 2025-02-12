import abc
from pathlib import Path
import pickle
from typing import Any, Dict, Sequence, Tuple

import mujoco
import numpy as np


class MujocoSceneTransform(abc.ABC):
    @abc.abstractmethod
    def apply(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        pass


class AddCameras(MujocoSceneTransform):
    def __init__(self, additional_cameras: Dict[str, Dict[str, Any]]):
        self.additional_cameras = additional_cameras

    def apply(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        metadata = {s.name: s.data for s in spec.texts}
        model_suffix = metadata.get('model_suffix')

        for camera_name, camera_cfg in self.additional_cameras.items():
            spec.worldbody.add_camera(name=f"{camera_name}{model_suffix}", pos=camera_cfg.pos, xyaxes=camera_cfg.xyaxes)

        return spec

class RecolorObject(MujocoSceneTransform):
    def __init__(self, object_name: str, color: list) -> None:
        super().__init__()
        self.object_name = object_name
        self.color = color

    def apply(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        # geom with name self.object_name
        geoms = [g for g in spec.geoms if g.name == self.object_name]
        assert len(geoms) == 1, f"Expected 1 geom with name {self.object_name}, found {len(geoms)}"
        geoms[0].rgba = np.array(self.color)

        return spec


def load_model_from_spec_file(xml_path: str, loaders: Sequence[MujocoSceneTransform] = ()) -> Tuple[mujoco.MjModel, Dict[str, str]]:
    spec, metadata = load_spec_from_file(xml_path, loaders)
    model = spec.compile()

    return model, metadata


def load_spec_from_file(xml_path: str, loaders: Sequence[MujocoSceneTransform] = ()) -> Tuple[mujoco.MjSpec, Dict[str, str]]:
    with open(xml_path, 'r') as f:
        xml_string = f.read()

    spec = mujoco.MjSpec.from_string(xml_string)

    metadata = {s.name: s.data for s in spec.texts}

    for loader in loaders:
        spec = loader.apply(spec)

    assets = metadata.get('relative_assets_path')

    if assets is not None:
        assets_path = Path(xml_path).parent / assets
        with open(assets_path, 'rb') as f:
            assets = pickle.load(f)
        spec.assets = assets

    return spec, metadata