import pytest
import time
import threading
from threading import Event
from control.ports import OutputPort, ThreadedInputPort, InputPortContainer, OutputPortContainer, DirectWriteInputPort, Empty
from control.world import MainThreadWorld

class MockWorld:
    def __init__(self):
        self.stop_event = Event()

    @property
    def should_stop(self):
        return self.stop_event.is_set()

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

    # Attempt to read from the input port - should raise StopIteration
    with pytest.raises(StopIteration):
        input_port.read(timeout=1)

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

def test_direct_write_input_port(world):
    port = DirectWriteInputPort(world)

    # Test writing and reading when world is running
    port.write(42, timestamp=100)
    result = port.read()
    assert result == (100, 42)

    # Test reading when empty
    assert port.read_nowait() is None

    # Test reading with timeout
    start_time = time.time()
    with pytest.raises(Empty):
        port.read(block=True, timeout=0.1)
    assert time.time() - start_time >= 0.1

    # Test reading when world is stopped
    world.stop_event.set()
    with pytest.raises(StopIteration):
        port.read()

    # Reset world state for the rest of the test
    world.stop_event.clear()

    # Test reading until stop
    port.write(1, timestamp=200)
    port.write(2, timestamp=300)

    def stop_world_after_delay():
        time.sleep(0.1)
        world.stop_event.set()

    stop_thread = threading.Thread(target=stop_world_after_delay)
    stop_thread.start()

    results = []
    try:
        for timestamp, value in port.read_until_stop():
            results.append((timestamp, value))
    except StopIteration:
        pass

    assert results == [(200, 1), (300, 2)]

    stop_thread.join()

def test_direct_write_input_port_subscribe(world):
    port = DirectWriteInputPort(world)

    received_values = []
    def callback(value, timestamp):
        received_values.append((timestamp, value))

    # Test subscription with context manager
    with port.subscribe(callback):
        port.write(42, timestamp=100)
        port.write(43, timestamp=101)

        # Values should not be in received_values yet because we haven't read them
        assert len(received_values) == 0

        # Read values - this should trigger callbacks
        result1 = port.read()
        result2 = port.read()

        assert result1 == (100, 42)
        assert result2 == (101, 43)

        # Check that callbacks were called during reads
        assert received_values == [(100, 42), (101, 43)]

    # Test that callback is unsubscribed after context exit
    received_values.clear()
    port.write(44, timestamp=102)
    port.read()  # Read the value
    assert len(received_values) == 0  # Callback should not receive values after unsubscribe

def test_threaded_input_port_subscribe(world):
    output_port = OutputPort(world)
    input_port = ThreadedInputPort(world, output_port)

    received_values = []
    def callback1(value, timestamp):
        received_values.append((1, timestamp, value))

    def callback2(value, timestamp):
        received_values.append((2, timestamp, value))

    # Test multiple subscriptions
    with input_port.subscribe(callback1, callback2):
        output_port.write("test1", timestamp=200)
        output_port.write("test2", timestamp=201)

        # Read to trigger callbacks
        input_port.read()
        input_port.read()

        # Verify both callbacks received values in order
        expected = [
            (1, 200, "test1"),
            (2, 200, "test1"),
            (1, 201, "test2"),
            (2, 201, "test2")
        ]
        assert received_values == expected

    # Test callbacks are unsubscribed
    received_values.clear()
    output_port.write("test3", timestamp=202)
    input_port.read()
    assert len(received_values) == 0

def test_input_port_container_with_subscriptions(world):
    output_container = OutputPortContainer(world, ["port1", "port2"], {})
    input_container = InputPortContainer(world, ["port1", "port2"], [])

    # Bind input ports to output ports
    input_container.port1 = output_container.port1
    input_container.port2 = output_container.port2

    received_values = []
    def callback(value, timestamp):
        received_values.append((timestamp, value))

    # Subscribe to one of the ports
    with input_container.port1.subscribe(callback):
        output_container.port1.write("value1", timestamp=300)
        output_container.port2.write("value2", timestamp=301)

        # Read from container with timeout
        start_time = time.time()
        results = []
        try:
            for result in input_container.read(timeout=0.1):
                if result[0] is not None:
                    results.append(result)
                if len(results) >= 2 or time.time() - start_time > 1.0:  # Safety timeout
                    world.stop_event.set()  # Signal stop properly
                    break
        finally:
            world.stop_event.set()  # Ensure cleanup

        # Check that callback received only port1's value
        assert received_values == [(300, "value1")]

        # Check that container read got both values
        assert len(results) == 2
        assert ("port1", 300, "value1") in results
        assert ("port2", 301, "value2") in results

if __name__ == "__main__":
    pytest.main([__file__])