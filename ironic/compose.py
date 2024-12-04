"""
Composition utilities for ironic control systems.

Example usage:
    source = DataSource()
    processor = Processor()
    sink = DataSink()

    # Simple composition with internal connections and properties
    system = compose(
        source,
        processor.bind(counter=source.outs.counter,
                      in_data=source.outs.data),
        sink.bind(data=processor.outs.processed,
                 counter_accumulated=processor.outs.counter_accumulated)
    )

    # Or with explicit input/output mapping:
    system = compose(
        components=[processor, sink],
        inputs={'data': (processor, 'in_data')},
        outputs={'result': (processor, 'processed')}
    )

    # Mappings can include None to skip certain ports:
    system = compose(
        components=[processor, sink],
        inputs={
            'data': (processor, 'in_data'),
            'optional_input': None  # This input will be skipped during binding
        },
        outputs={
            'result': (processor, 'processed'),
            'optional_output': None  # This output will be skipped during connection
        }
    )
"""

import asyncio
from typing import Dict, Sequence, Any, Tuple, Optional
from types import SimpleNamespace

from .system import ControlSystem, ironic_system, State


class CompositionError(Exception):
    """Error raised when composition validation fails"""
    pass


@ironic_system(input_ports=[], output_ports=[])
class ComposedSystem(ControlSystem):
    """A control system composed of other control systems"""
    def __init__(self,
                 components: Sequence[ControlSystem],
                 inputs: Optional[Dict[str, Tuple[ControlSystem, str]]] = None,
                 outputs: Optional[Dict[str, Tuple[ControlSystem, str]]] = None):
        components_set = set(components)

        # Validate all referenced components exist in composition
        for mapping_type, mappings in [('Input', inputs), ('Output', outputs)]:
            if mappings and any(comp[0] not in components_set for comp in mappings.values() if comp is not None):
                raise CompositionError(f"{mapping_type} mappings reference components not in composition")

        # Get input/output ports from mappings
        input_ports = list(inputs.keys()) if inputs else []
        output_ports = list(outputs.keys()) if outputs else []

        # Update class-level port definitions
        self.__class__._input_ports = input_ports
        self.__class__._output_ports = output_ports

        super().__init__()
        self._components = components

        # Store the component and port name for each input
        self._input_mappings = dict(inputs or {})

        # Connect outputs - direct assignment of OutputPort objects
        if outputs:
            for name, binding in outputs.items():
                if binding is None:
                    continue
                component, port_name = binding
                try:
                    original_port = getattr(component.outs, port_name)
                except AttributeError:
                    raise CompositionError(f"Component {component} does not have output port '{port_name}'")
                setattr(self.outs, name, original_port)

    async def setup(self):
        """Set up all components"""
        for component in self._components:
            await component.setup()

    async def cleanup(self):
        """Clean up all components"""
        for component in self._components:
            await component.cleanup()

    async def step(self):
        """Step all components"""
        results = await asyncio.gather(*[component.step() for component in self._components])
        return State.ALIVE if all(result == State.ALIVE for result in results) else State.FINISHED

    def bind(self, **bindings):
        """Bind inputs to the appropriate components"""
        binds = {}

        for name, binding in bindings.items():
            if name not in self._input_mappings:
                raise ValueError(f"Unknown input: {name}")

            input_mapping = self._input_mappings[name]
            if input_mapping is None:
                continue
            component, port_name = input_mapping
            component.bind(**{port_name: binding})
            binds[name] = binding

        self.ins = SimpleNamespace(**binds)
        return self


def compose(*components: ControlSystem,
           inputs: Dict[str, Tuple[ControlSystem, str]] = None,
           outputs: Dict[str, Tuple[ControlSystem, str]] = None) -> ComposedSystem:
    """
    Compose multiple control systems into a single system.

    The composed system exposes selected inputs and outputs that can be bound to other systems.
    This allows hierarchical composition where composed systems can be treated as regular
    control systems and connected to other systems.

    Args:
        *components: Variable number of control system instances to compose
        inputs: Dictionary mapping external input names to tuples of (component, port_name)
               These inputs can later be bound to outputs of other systems
        outputs: Dictionary mapping external output names to tuples of (component, port_name)
                These outputs can be bound to inputs of other systems

    Returns:
        A composed control system instance that can be bound to other systems

    Example:
        source = DataSource()
        processor = Processor()
        sink = DataSink()

        # Simple composition with internal connections
        system = compose(
            source,
            processor.bind(counter=source.outs.counter,
                           in_data=source.outs.data),
            sink.bind(data=processor.outs.processed,
                      counter_accumulated=processor.outs.counter_accumulated)
        )

        # Composition with exposed ports that can be bound to other systems:
        subsystem = compose(
            processor,
            sink,
            inputs={'data': (processor, 'in_data')},
            outputs={'result': (processor, 'processed')}
        )

        # The composed subsystem can be bound to other systems:
        other_system = compose(
            source,
            subsystem.bind(data=source.outs.data)  # Binding exposed input to source output
        )
    """
    return ComposedSystem(
        components=list(components),  # Convert tuple to list
        inputs=inputs,
        outputs=outputs
    )