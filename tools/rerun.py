# Control system for a logging tool rerun

import asyncio
from typing import Dict, Optional, Callable, Any

import numpy as np
import rerun as rr

from control import ControlSystem, World


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


class Rerun(ControlSystem):
    def __init__(self, world: World, recording_id: str, spawn=False, save_path=None, connect=None, inputs: Dict[str, Optional[Callable[[Any], Any]]] = None):
        rr.init(recording_id, spawn=spawn)
        if save_path is not None:
            rr.save(save_path)
        if connect is not None:
            rr.connect(connect)
        super().__init__(world, inputs=inputs.keys(), outputs=[])
        self._input_fns = inputs

    def _default_fn(self, data: Any) -> Any:
        if isinstance(data, (list, np.ndarray)):
            return log_array
        elif isinstance(data, (int, float)):
            return log_scalar

        return None

    def _log(self, name: str, ts: int, data: Any):
        fn = self._input_fns[name]
        if fn is None:
            fn = self._default_fn(data)
        fn(name, ts, data)

    def run(self):
        try:
            for name, ts, data in self.ins.read():
                self._log(name, ts, data)
        finally:
            rr.disconnect()
