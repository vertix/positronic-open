import pytest
from control.system import ControlSystem, control_system
from control.world import MainThreadWorld
from control.ports import InputPort, OutputPort

# Test basic decorator configurations
def test_empty_control_system():
    @control_system(inputs=[], outputs=[])
    class EmptySystem(ControlSystem):
        def run(self):
            pass

    world = MainThreadWorld()
    system = EmptySystem(world)

    assert hasattr(system, 'ins')
    assert hasattr(system, 'outs')
    assert system.ins.size() == 0
    assert system.outs.size() == 0

def test_inputs_only_system():
    @control_system(inputs=["input1", "input2"])
    class InputsSystem(ControlSystem):
        def run(self):
            pass

    world = MainThreadWorld()
    system = InputsSystem(world)

    assert system.ins.size() == 2
    assert system.ins.input1 is not None
    assert system.ins.input2 is not None
    assert system.outs.size() == 0

def test_outputs_only_system():
    @control_system(outputs=["output1", "output2"])
    class OutputsSystem(ControlSystem):
        def run(self):
            pass

    world = MainThreadWorld()
    system = OutputsSystem(world)

    assert system.ins.size() == 0
    assert system.outs.size() == 2
    assert system.outs.output1 is not None
    assert system.outs.output2 is not None

def test_lazy_port_initialization():
    @control_system(inputs=["input1"], outputs=["output1"])
    class LazySystem(ControlSystem):
        def run(self):
            pass

    world = MainThreadWorld()
    system = LazySystem(world)

    # Check that _inputs and _outputs don't exist before access
    assert not hasattr(system, '_inputs')
    assert not hasattr(system, '_outputs')

    # Access ports to trigger initialization
    _ = system.ins.input1
    _ = system.outs.output1

    # Now they should exist
    assert hasattr(system, '_inputs')
    assert hasattr(system, '_outputs')

def test_port_types():
    @control_system(inputs=["input1"], outputs=["output1"])
    class TypeSystem(ControlSystem):
        def run(self):
            pass

    world = MainThreadWorld()
    system = TypeSystem(world)

    assert isinstance(system.ins.input1, InputPort)
    assert isinstance(system.outs.output1, OutputPort)

def test_system_world_registration():
    @control_system(inputs=["input1"])
    class WorldSystem(ControlSystem):
        def run(self):
            pass

    world = MainThreadWorld()
    system = WorldSystem(world)

    # Verify that the system is properly initialized with the world
    assert system.world == world
    assert not world.should_stop  # Verify initial state


if __name__ == "__main__":
    pytest.main([__file__])