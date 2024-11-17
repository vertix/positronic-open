import pytest
from ironic import ControlSystem, Message, ironic_system, on_message, out_property, OutputPort


class TestSystemBase(ControlSystem):
    """Base class to avoid pytest warnings about __init__"""
    def __init__(self):
        super().__init__()
        self.received_messages = []
        self.config_value = None


@ironic_system(
    input_ports=['sensor'],
    output_ports=['processed'],
    input_props=['config'],
    output_props=['status']
)
class TestSystem(TestSystemBase):
    @on_message('sensor')
    async def handle_sensor(self, message: Message):
        self.received_messages.append(message)
        # Echo message to processed output with timestamp
        await self.outs.processed.write(Message(
            data=f"processed_{message.data}",
            timestamp=message.timestamp
        ))

    @out_property()
    async def status(self):
        return "ok", 123


@pytest.mark.asyncio
async def test_output_port_subscription():
    """Test that output port subscriptions work."""
    system = TestSystem()
    received_messages = []

    # Subscribe to the output port with an async lambda
    async def handler(msg):
        received_messages.append(msg)
    system.outs.processed.subscribe(handler)

    # Send a message through input port handler
    test_message = Message(data="test", timestamp=1)
    await system.handle_sensor(test_message)

    assert len(received_messages) == 1
    assert received_messages[0].data == "processed_test"
    assert received_messages[0].timestamp == 1


@pytest.mark.asyncio
async def test_binding():
    """Test that binding inputs to outputs works."""
    system1 = TestSystem()
    system2 = TestSystem()

    # Create a test output port
    test_port = OutputPort("test")

    # Bind system1's sensor input to test port
    system1.bind(sensor=test_port)
    # Bind system2's sensor input to system1's processed output
    system2.bind(sensor=system1.outs.processed)

    # Send message through the test port
    test_message = Message(data="test", timestamp=1)
    await test_port.write(test_message)

    # Check that system1 received the original message
    assert len(system1.received_messages) == 1
    assert system1.received_messages[0].data == "test"
    assert system1.received_messages[0].timestamp == 1

    # Check that system2 received the processed message
    assert len(system2.received_messages) == 1
    assert system2.received_messages[0].data == "processed_test"
    assert system2.received_messages[0].timestamp == 1


def test_invalid_binding():
    """Test that binding invalid inputs raises error."""
    system = TestSystem()

    with pytest.raises(ValueError, match="Unknown input: invalid_port"):
        system.bind(invalid_port=None)


def test_double_binding():
    """Test that binding inputs twice raises error."""
    system1 = TestSystem()
    system2 = TestSystem()

    system2.bind(sensor=system1.outs.processed)

    with pytest.raises(AssertionError, match="Inputs already bound"):
        system2.bind(sensor=system1.outs.processed)


class PropertySystemBase(ControlSystem):
    """Base class to avoid pytest warnings about __init__"""
    def __init__(self):
        super().__init__()
        self._multiplier = 1


@ironic_system(
    input_props=['multiplier'],
    output_props=['value']
)
class PropertySystem(PropertySystemBase):
    @out_property()
    async def value(self):
        return self._multiplier * 10, 123

    def set_multiplier(self, value):
        self._multiplier = value


@pytest.mark.asyncio
async def test_property_binding():
    """Test that property binding works."""
    system1 = PropertySystem()
    system2 = PropertySystem()

    # Bind system2's multiplier to system1's value
    system2.bind(multiplier=system1.value)

    # Get initial value
    value, ts = await system2.ins.multiplier()
    assert value == 10  # default multiplier = 1
    assert ts == 123

    # Change system1's multiplier
    system1._multiplier = 2

    # Get updated value
    value, ts = await system2.ins.multiplier()
    assert value == 20
    assert ts == 123