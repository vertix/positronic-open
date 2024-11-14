import pytest
import time
import threading
from threading import Event
from control.ports import InputPort, OutputPort, ThreadedInputPort, InputPortContainer, OutputPortContainer, DirectWriteInputPort
from control.world import MainThreadWorld
from control import utils


class MockWorld:
    def __init__(self):
        self.stop_event = Event()

@pytest.fixture
def world():
    return MockWorld()

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
    output_container = OutputPortContainer(world, ["port1", "port2"], {})
    input_container = InputPortContainer(world, ["port1", "port2"], [])

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
    output_container = OutputPortContainer(world, ["port1", "port2"], {})
    input_container = InputPortContainer(world, ["port1", "port2"], [])

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

def test_direct_write_input_port():
    world = MainThreadWorld()
    port = DirectWriteInputPort(world)

    # Test writing and reading
    port.write(42, timestamp=100)
    result = port.read()
    assert result == (100, 42)

    # Test reading when empty
    assert port.read_nowait() is None

    # Test reading with timeout
    start_time = time.time()
    assert port.read(block=True, timeout=0.1) is None
    assert time.time() - start_time >= 0.1

    # Test reading until stop
    port.write(1, timestamp=200)
    port.write(2, timestamp=300)

    def stop_world_after_delay():
        time.sleep(0.1)
        world.stop_event.set()

    stop_thread = threading.Thread(target=stop_world_after_delay)
    stop_thread.start()

    results = list(port.read_until_stop())
    assert results == [(200, 1), (300, 2)]

    stop_thread.join()

def test_map_port(world):
    output_port = OutputPort(world)
    input_port = InputPortContainer(world, ["input"], [])

    @utils.map_port
    def square(value):
        return value * value

    input_port.bind(input=square(output_port))

    output_port.write(2, timestamp=0)
    assert input_port.input.read() == (0, 4)


def test_port_to_prop():
    world = MainThreadWorld()
    output_port = OutputPort(world)
    prop = utils.port_to_prop(output_port)
    world.run()
    time.sleep(0.2)

    output_port.write(2, timestamp=0)
    time.sleep(0.2)
    assert prop() == (0, 2)

if __name__ == "__main__":
    pytest.main([__file__])