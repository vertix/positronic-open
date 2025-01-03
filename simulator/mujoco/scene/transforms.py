import abc
from typing import Any, Dict, Optional, Sequence

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

        for camera_name, camera_cfg in self.additional_cameras.items():
            spec.worldbody.add_camera(name=camera_name, pos=camera_cfg.pos, xyaxes=camera_cfg.xyaxes)

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


def load_model_from_spec(xml_string: str, assets: Optional[Dict[str, Any]] = None, loaders: Sequence[MujocoSceneTransform] = ()) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_string(xml_string)

    for loader in loaders:
        spec = loader.apply(spec)

    if assets is None:
        return spec.compile()

    return spec.compile(assets)
