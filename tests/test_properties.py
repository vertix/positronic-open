import time
from typing import List, Tuple

import numpy as np
import pytest

from control.system import ControlSystem, control_system, output_property
from control.world import MainThreadWorld, World


@pytest.fixture
def world() -> MainThreadWorld:
    return MainThreadWorld()


@control_system(output_props=["output"])
class OutputTestSystem(ControlSystem):
    def __init__(self, world: World):
        super().__init__(world)
        self._counter = 0.0

    @output_property("output")
    def output(self):
        self._counter += 1.0
        return self._counter, self.world.now_ts

    def run(self):
        time.sleep(1)


@control_system(input_props=["input"])
class InputTestSystem(ControlSystem):
    def __init__(self, world: World, out_container: List[Tuple[int, float]]):
        super().__init__(world)
        self.output = out_container

    def run(self):
        for i in range(5):
            self.output.append(self.ins.input())


def test_properties(world: MainThreadWorld):
    output = []
    input_system = InputTestSystem(world, output)
    output_system = OutputTestSystem(world)

    input_system.ins.input = output_system.outs.output

    world.run()

    vals, ts = zip(*output)
    assert vals == (1.0, 2.0, 3.0, 4.0, 5.0)
    assert all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1)), "Timestamps are not monotonic"


if __name__ == "__main__":
    pytest.main([__file__])
