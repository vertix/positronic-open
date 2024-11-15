import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from control.world import World
from .system import ControlSystem, control_system, output_property, output_property_custom_time
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


def map_prop(fn: Callable[[Any], Any]):
    def _wrapper(original):
        def _internal():
            value, ts = original()
            return fn(value), ts
        return _internal
    return _wrapper


def control_system_fn(*, inputs: List[str] = None, outputs: List[str] = None, input_props: List[str] = None):
    def decorator(fn):
        @control_system(inputs=inputs, outputs=outputs, input_props=input_props)
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

@control_system(output_props=["prop_values"])
class PropDict(ControlSystem):
    def __init__(self, world: World, input_props: Dict[str, Callable[[], Any]]):
        super().__init__(world)
        self.input_props = input_props

    @output_property_custom_time("prop_values")
    def prop_values(self):
        prop_times = {}
        prop_values = {}
        for name, fn in self.input_props.items():
            value, ts = fn()
            prop_values[name] = value
            prop_times[name] = ts
        
        time_list = list(prop_times.values())
        # warn if time range is too large
        if len(time_list) > 0:
            time_range = max(time_list) - min(time_list)
            if time_range > 10:
                print(f"Warning: time range for prop_values is {time_range} ms")
        
        return prop_values, min(time_list)

    def run(self):
        pass
