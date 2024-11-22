import pytest
import asyncio
from typing import List, Tuple

import ironic as ir
from ironic.compose import ComposedSystem, CompositionError, compose


@ir.ironic_system(output_ports=['data'], output_props=['counter'])
class DataSource(ir.ControlSystem):
    def __init__(self):
        super().__init__()
        self._counter = 0

    @ir.out_property
    async def counter(self):
        return ir.Message(self._counter)

    async def step(self):
        await self.outs.data.write(ir.Message(self._counter))
        self._counter += 1
        return ir.State.ALIVE


@ir.ironic_system(input_ports=['in_data'],
                  input_props=['counter'],
                  output_ports=['processed'],
                  output_props=['counter_accumulated'])
class Processor(ir.ControlSystem):
    def __init__(self):
        super().__init__()
        self.received: List[int] = []
        self._counter_accumulated = 0

    @ir.out_property
    async def counter_accumulated(self):
        return ir.Message(self._counter_accumulated)

    @ir.on_message('in_data')
    async def on_data(self, message: ir.Message):
        counter_msg = await self.ins.counter()
        self._counter_accumulated += counter_msg.data

        self.received.append(message.data)
        await self.outs.processed.write(ir.Message(message.data * 2))



@ir.ironic_system(input_ports=['data'], input_props=['counter_accumulated'])
class DataSink(ir.ControlSystem):
    def __init__(self):
        super().__init__()
        self.received: List[Tuple[int, int]] = []

    @ir.on_message('data')
    async def on_data(self, message: ir.Message):
        counter_msg = await self.ins.counter_accumulated()
        self.received.append((message.data, counter_msg.data))


@pytest.mark.asyncio
async def test_simple_composition():
    """Test composing systems with internal connections"""
    source = DataSource()
    processor = Processor()
    sink = DataSink()

    # Compose all systems
    system = compose(
        source,
        processor.bind(counter=source.outs.counter,
                      in_data=source.outs.data),
        sink.bind(data=processor.outs.processed,
                 counter_accumulated=processor.outs.counter_accumulated)
    )

    await system.setup()

    # Run for a few steps
    for _ in range(3):
        await system.step()

    await system.cleanup()

    # Check that data flowed through all components
    assert processor.received == [0, 1, 2]
    assert sink.received == [(0, 0), (2, 1), (4, 3)]  # Doubled values and accumulated counter


@pytest.mark.asyncio
async def test_exposed_outputs():
    """Test composing systems with exposed outputs"""
    source = DataSource()
    processor = Processor()

    # Compose with exposed output
    system = compose(
        source,
        processor.bind(counter=source.counter,
                      in_data=source.outs.data),
        outputs={
            'result': (processor, 'processed'),
            'counter': (processor, 'counter_accumulated')
        }
    )

    # Create external sink
    sink = DataSink().bind(data=system.outs.result,
                           counter_accumulated=system.outs.counter)

    await system.setup()
    await sink.setup()

    # Run for a few steps
    for _ in range(3):
        await system.step()
        await sink.step()

    await sink.cleanup()
    await system.cleanup()

    assert processor.received == [0, 1, 2]
    assert sink.received == [(0, 0), (2, 1), (4, 3)]


@pytest.mark.asyncio
async def test_exposed_inputs():
    """Test composing systems with exposed inputs"""
    processor = Processor()
    sink = DataSink()

    # Compose with exposed input
    system = compose(
        processor,
        sink.bind(data=processor.outs.processed,
                 counter_accumulated=processor.outs.counter_accumulated),
        inputs={
            'data': (processor, 'in_data'),
            'counter': (processor, 'counter')
        }
    )

    # Create external source
    source = DataSource()

    # Connect composed system to source
    system.bind(data=source.outs.data, counter=source.outs.counter)

    await source.setup()
    await system.setup()

    # Run for a few steps
    for _ in range(3):
        await source.step()
        await system.step()

    await source.cleanup()
    await system.cleanup()

    assert processor.received == [0, 1, 2]
    assert sink.received == [(0, 0), (2, 1), (4, 3)]


@pytest.mark.asyncio
async def test_invalid_input():
    """Test error handling for invalid inputs"""
    processor = Processor()
    system = compose(
        processor,
        inputs={'valid_input': (processor, 'in_data')}
    )

    with pytest.raises(ValueError, match="Unknown input: invalid"):
        system.bind(invalid=None)


@pytest.mark.asyncio
async def test_invalid_component_references():
    """Test validation of component references in inputs/outputs"""
    source = DataSource()
    processor = Processor()
    other_processor = Processor()  # Not included in composition

    # Test invalid input reference
    with pytest.raises(CompositionError, match="Input mappings reference components not in composition"):
        compose(
            source,
            processor,
            inputs={'data': (other_processor, 'in_data')}  # other_processor not in components
        )

    # Test invalid output reference
    with pytest.raises(CompositionError, match="Output mappings reference components not in composition"):
        compose(
            source,
            processor,
            outputs={'result': (other_processor, 'processed')}  # other_processor not in components
        )

    # Valid composition should work
    system = compose(
        source,
        processor,
        inputs={'data': (processor, 'in_data')},
        outputs={'result': (processor, 'processed')}
    )
    assert isinstance(system, ComposedSystem)
