import time
from typing import List, Tuple

import numpy as np
import pytest

from control.ports import output_property
from control.system import ControlSystem, control_system
from control.world import MainThreadWorld, World


@pytest.fixture
def world() -> MainThreadWorld:
    return MainThreadWorld()


@control_system(output_props=["output"])
class OutputTestSystem(ControlSystem):
    def __init__(self, world: World):
        super().__init__(world)
        self._counter = 0.0

    @output_property
    def output(self):
        self._counter += 1.0
        return self.world.now_ts, self._counter

    def run(self):
        while not self.should_stop:
            time.sleep(1)


@control_system(input_props=["input"])
class InputTestSystem(ControlSystem):
    def __init__(self, world: World, output: List[Tuple[int, float]]):
        super().__init__(world)
        self.output = output

    def run(self):
        for i in range(5):
            self.output.append(self.ins.input())


def test_properties(world: MainThreadWorld):
    output = []
    input_system = InputTestSystem(world, output)
    output_system = OutputTestSystem(world)

    input_system.ins.input = output_system.outs.output

    world.run()

    ts, vals = zip(*output)
    assert all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1)), "Timestamps are not monotonic"
    assert vals == tuple(range(5))


if __name__ == "__main__":
    pytest.main([__file__])
