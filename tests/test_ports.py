import pytest
import time
import threading
from threading import Event
from control.ports import OutputPort, ThreadedInputPort, InputPortContainer, OutputPortContainer

class MockWorld:
    def __init__(self):
        self.stop_event = Event()

@pytest.fixture
def world():
    return MockWorld()

class TestThreadedInputPort:
    @pytest.fixture
    def output_port(self, world):
        return OutputPort(world)

    @pytest.fixture
    def input_port(self, world, output_port):
        return ThreadedInputPort(world, output_port)

    def test_last_property_returns_last_value(self, input_port, output_port):
        output_port.write("first", 1)
        output_port.write("second", 2)
        output_port.write("third", 3)
        time.sleep(0.1)  # Allow time for the writes to process
        assert input_port.last == (3, "third")
        assert input_port.read_nowait() is None  # All values should be consumed

    def test_last_property_returns_none_when_empty(self, input_port):
        assert input_port.last is None

    def test_last_property_with_interleaved_reads(self, input_port, output_port):
        output_port.write("first", 1)
        output_port.write("second", 2)
        time.sleep(0.1)  # Allow time for the writes to process
        assert input_port.read_nowait() == (1, "first")
        output_port.write("third", 3)
        time.sleep(0.1)  # Allow time for the write to process
        assert input_port.last == (3, "third")
        assert input_port.read_nowait() is None  # All values should be consumed

    def test_read_after_last(self, input_port, output_port):
        output_port.write("first", 1)
        output_port.write("second", 2)
        time.sleep(0.1)  # Allow time for the writes to process
        assert input_port.last == (2, "second")
        output_port.write("third", 3)
        time.sleep(0.1)  # Allow time for the write to process
        assert input_port.read_nowait() == (3, "third")
        assert input_port.read_nowait() is None  # All values should be consumed

def test_threaded_input_port_respects_stop_event(world):
    output_port = OutputPort(world)
    input_port = ThreadedInputPort(world, output_port)

    # Write a value to the output port
    output_port.write("test_value")

    # Set the stop event
    world.stop_event.set()

    # Attempt to read from the input port
    result = input_port.read(timeout=1)

    # The read should return None due to the stop event
    assert result is None

def test_input_port_container_respects_stop_event(world):
    output_container = OutputPortContainer(world, ["port1", "port2"])
    input_container = InputPortContainer(world, ["port1", "port2"])

    # Bind input ports to output ports
    input_container.port1 = output_container.port1
    input_container.port2 = output_container.port2

    # Write values to output ports
    output_container.port1.write("value1")
    output_container.port2.write("value2")

    # Set the stop event
    world.stop_event.set()

    # Attempt to read from the input container
    results = list(input_container.read(timeout=1))

    # The read should return an empty list due to the stop event
    assert len(results) == 0

def test_input_port_container_read_with_timeout(world):
    output_container = OutputPortContainer(world, ["port1", "port2"])
    input_container = InputPortContainer(world, ["port1", "port2"])

    # Bind input ports to output ports
    input_container.port1 = output_container.port1
    input_container.port2 = output_container.port2

    # Write a value to one output port
    output_container.port1.write("value1")

    # Read from the input container with a short timeout
    results = []
    for i, result in enumerate(input_container.read(timeout=0.1)):
        if i >= 1:
            world.stop_event.set()
        results.append(result)

    # We should get one value and then a None
    assert len(results) == 2
    assert results[0] == ("port1", None, "value1")
    assert results[1] == (None, None, None)

if __name__ == "__main__":
    pytest.main([__file__])