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
from typing import Callable, Dict, Sequence, Set, Tuple, Optional
from types import SimpleNamespace

from .system import ControlSystem, OutputPort, ironic_system, State


class CompositionError(Exception):
    """Error raised when composition validation fails"""
    pass

def _gather_components(components: Sequence[ControlSystem]) -> Set[ControlSystem]:
    """
    Recursively gather all components from a sequence of control systems

    Args:
        components: A sequence of control systems to gather components from

    Returns:
        A set of all components in the composition
    """
    res = set()

    for component in components:
        res.add(component)

        if isinstance(component, ComposedSystem):
            res.update(_gather_components(component._components))

    return res


@ironic_system(input_ports=[], output_ports=[])
class ComposedSystem(ControlSystem):
    """A control system composed of other control systems"""
    def __init__(self,
                 components: Sequence[ControlSystem],
                 inputs: Optional[Dict[str, Tuple[ControlSystem, str]]] = None,
                 outputs: Optional[Dict[str, OutputPort | Callable]] = None):
        components_set = _gather_components(components)

        # Validate all referenced components exist in composition
        if inputs and any(comp[0] not in components_set for comp in inputs.values() if comp is not None):
            raise CompositionError(f"Input mappings reference components not in composition")

        if outputs:
            for name, out in outputs.items():
                if out is None:
                    continue

                if isinstance(out, OutputPort):
                    if out.parent_system is not None and out.parent_system not in components_set:
                        raise CompositionError(f"Output mappings reference components not in composition. Output: '{name}'. Parent system: {out.parent_system}")
                elif hasattr(out, '__self__'):
                    if out.__self__ not in components_set:
                        raise CompositionError(f"Output mappings reference components not in composition. Output: '{name}'. Parent system: {out.__self__}")

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
            for name, original_port in outputs.items():
                if original_port is None:
                    continue
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
           outputs: Dict[str, OutputPort | Callable] = None) -> ComposedSystem:
    """
    Compose multiple control systems into a single system.

    The composed system exposes selected inputs and outputs that can be bound to other systems.
    This allows hierarchical composition where composed systems can be treated as regular
    control systems and connected to other systems.

    Args:
        *components: Variable number of control system instances to compose
        inputs: Dictionary mapping external input names to tuples of (component, port_name)
               These inputs can later be bound to outputs of other systems
        outputs: Dictionary mapping external output names to output ports or properties.
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
            outputs={'result': processor.outs.processed}
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

def extend(system: ControlSystem, outputs: Dict[str, OutputPort | Callable]) -> ComposedSystem:
    """
    Extend a control system with additional outputs.

    This function allows adding new outputs to an existing control system without
    modifying its existing structure. It returns a new composed system that includes
    both the original system and the new outputs.

    Args:
        system: The control system to extend
        outputs: Dictionary mapping additional output names to output ports or properties

    Returns:
        A new composed control system instance that includes both the original system
        and the new outputs

    Example:
        >>> system = DataSource()
        >>> extended = extend(system, {'value_squared': ir.utils.map_property(lambda x: x ** 2, system.outs.value)})
    """
    original_inputs = {}
    for input_port in system._input_ports:
        original_inputs[input_port] = (system, input_port)

    original_outputs = {name: getattr(system.outs, name) for name in system.outs.__dict__.keys()}
    original_outputs.update(outputs)

    return ComposedSystem(
        components=[system],
        inputs=original_inputs,
        outputs=original_outputs
    )
