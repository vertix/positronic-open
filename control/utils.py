import logging
import time
from typing import Any, Callable, List, Optional

from control.world import World
from .system import ControlSystem, control_system
from .ports import OutputPort
from .ports import InputPort


def map_port(fn: Callable[[Any], Any]):
    class _MapPort(OutputPort):
        def __init__(self, original: OutputPort):
            super().__init__(original.world)
            self.original = original
            self.original._bind(self.write)

        def write(self, value: Any, timestamp: Optional[int] = None):
            super().write(fn(value), timestamp)

    return _MapPort

# TODO: Add world as an argument to the internal function.
def control_system_fn(*, inputs: List[str] = None, outputs: List[str] = None):
    def decorator(fn):
        @control_system(inputs=inputs, outputs=outputs)
        class _ControlSystem(ControlSystem):
            def run(self):
                fn(self.ins, self.outs)

        return _ControlSystem

    return decorator

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
