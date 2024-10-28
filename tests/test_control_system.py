import pytest
from control.utils import control_system_fn
from control.world import MainThreadWorld
from control.ports import DirectWriteInputPort, InputPort, OutputPort

def test_control_system_decorator():
    @control_system_fn(inputs=["input1", "input2"], outputs=["output1"])
    def my_system(ins, outs):
        _, value1 = ins.input1.read()
        _, value2 = ins.input2.read()
        value = value1 + value2
        outs.output1.write(value)

    world = MainThreadWorld()
    system = my_system(world)

    out_port = DirectWriteInputPort(world)
    system.outs.output1._bind(out_port.write)

    # Check if the system has the correct inputs and outputs
    assert isinstance(system.ins.input1, InputPort)
    assert isinstance(system.ins.input2, InputPort)
    assert isinstance(system.outs.output1, OutputPort)

    # Test the system's functionality
    system.ins.input1.write(5)
    system.ins.input2.write(3)
    system.run()

    result = out_port.read()
    assert result == (None, 8)


if __name__ == "__main__":
    pytest.main([__file__])