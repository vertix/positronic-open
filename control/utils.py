import time
from typing import Any, Callable, List, Optional
from .system import ControlSystem, OutputPort
import logging


def map_port(fn: Callable[[Any], Any]):
    class _MapOutputPort(OutputPort):
        def __init__(self, original: OutputPort):
            super().__init__()
            self.original = original
            self.original.bound_to.append(self)

        async def write(self, value: Any, timestamp: Optional[int] = None):
            await super().write(fn(value), timestamp)

    return _MapOutputPort


class FPSCounter:
    def __init__(self, prefix: str, report_every_sec: float = 10.0):
        self.prefix = prefix
        self.report_every_sec = report_every_sec
        self.last_report_time = time.monotonic()
        self.frame_count = 0

    def tick(self):
        self.frame_count += 1
        if time.monotonic() - self.last_report_time >= self.report_every_sec:
            fps = self.frame_count / (time.monotonic() - self.last_report_time)
            print(f"{self.prefix}: {fps:.2f} fps")
            self.last_report_time = time.monotonic()
            self.frame_count = 0

class Logger(ControlSystem):
    """
    Logger is a System that logs the values of its input ports.
    """
    def __init__(self, level: int = logging.INFO, inputs: List[str] = []):
        super().__init__(inputs=inputs)
        self.level = level

    async def run(self):
        async for name, _ts, value in self._inputs.read():
            logging.log(self.level, f"{name}: {value}")


class MapSystem(ControlSystem):
    """
    Map is a System that applies a mapping function to its input ports and writes the result to
    the corresponding output ports. The mapping function can be specified for each input port
    individually via keyword arguments, or a default mapping function can be provided. Either a
    default mapping function must be defined, or all mappings for all inputs must be defined.

    Example usage:
    map = Map(inputs=['joints', 'pos'],
              default_map_fn=lambda n, v: v * 2,
              joints=lambda n, v: set_joints(v))
    """
    def __init__(self, inputs: List[str], default: Callable[[str, Any], Any] = None, **kwargs):
        """
        Initialize the Map system. The outputs have the same names as the inputs.
        """
        super().__init__(inputs=inputs, outputs=inputs)
        self.default_map_fn = default
        self.map_fns = kwargs

        if not self.default_map_fn and not all(input_name in self.map_fns for input_name in inputs):
            raise ValueError("Either default_map_fn must be defined, or all mappings for all inputs must be defined")

    async def run(self):
        """
        Control loop coroutine.
        """
        async for name, _ts, value in self._inputs.read():
            map_fn = self.map_fns.get(name, self.default_map_fn)
            await self.outs[name].write(map_fn(name, value))
