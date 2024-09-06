# Control system for a logging tool rerun

import asyncio
from typing import Dict, Optional, Callable, Any

import numpy as np
import rerun as rr

from control import ControlSystem, World
import geom


def log_array(name: str, ts: int, data: np.ndarray):
    rr.set_time_seconds('time', ts / 1000)
    for i, d in enumerate(data):
        rr.log(f"{name}/{i}", rr.Scalar(d))

def log_scalar(name: str, ts: int, data: float):
    rr.set_time_seconds('time', ts / 1000)
    rr.log(name, rr.Scalar(data))

def log_image(name: str, ts: int, data: np.ndarray):
    rr.set_time_seconds('time', ts / 1000)
    rr.log(name, rr.Image(data))


def log_transform(name: str, ts: int, data: geom.Transform3D):
    rr.set_time_seconds('time', ts / 1000)
    vectors = np.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.1]])
    for i, v in enumerate(vectors):
        vectors[i] = data.quaternion(v)
    rr.log(name, rr.Arrows3D(origins=np.repeat(data.translation, 3), vectors=vectors,
                             colors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])))


class Rerun(ControlSystem):
    def __init__(self, world: World, recording_id: str, spawn=False, save_path=None, connect=None, inputs: Dict[str, Optional[Callable[[Any], Any]]] = None):
        rr.init(recording_id, spawn=spawn)
        self.save_path = save_path
        self.connect = connect
        super().__init__(world, inputs=inputs.keys(), outputs=[])
        self._input_fns = inputs

    def _default_fn(self, data: Any, name: str) -> Any:
        if isinstance(data, (list, np.ndarray)):
            if data.ndim == 1:
                return log_array
            elif data.ndim == 0:
                return log_scalar
            elif data.ndim in (2, 3):
                return log_image
        elif isinstance(data, (int, float)):
            return log_scalar
        elif isinstance(data, geom.Transform3D):
            return log_transform

        raise ValueError(f"Unsupported data type: {type(data)} for port {name}")

    def _log(self, name: str, ts: int, data: Any):
        fn = self._input_fns[name]
        if fn is None:
            fn = self._default_fn(data, name)
        try:
            fn(name, ts, data)
        except:
            print(f'Unable to log {name} at {ts}')
            raise

    def run(self):
        if self.save_path is not None:
            rr.save(self.save_path)
        elif self.connect is not None:
            rr.connect(self.connect)
        try:
            for name, ts, data in self.ins.read():
                self._log(name, ts, data)
        finally:
            rr.disconnect()
