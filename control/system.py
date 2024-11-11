from abc import ABC, abstractmethod
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


from control.ports import InputPortContainer, OutputPortContainer


def control_system(*, inputs: List[str] = None, outputs: List[str] = None,
                   input_props: List[str] = None, output_props: List[str] = None):
    """
    Class decorator for defining ports on a control system.

    Ports are push communication medium (similar to channels).
    Properties are pull communication medium (similar to RPC).
    Both ports and properties return [value, timestamp] tuple as their results.
    """
    if inputs is None:
        inputs = []
    if outputs is None:
        outputs = []
    if input_props is None:
        input_props = []
    if output_props is None:
        output_props = []

    def decorator(cls):
        # Store port definitions on class
        cls._input_ports = inputs
        cls._output_ports = outputs
        cls._input_props = input_props
        cls._output_props = output_props

        # Check output properties at class definition time
        defined_props = {
            method.__output_property_name__
            for method in cls.__dict__.values()
            if hasattr(method, '__is_output_property__')
        }
        missing_props = set(output_props) - defined_props
        if missing_props:
            raise ValueError(f"Output properties {missing_props} are not defined in class {cls.__name__}")

        # Add lazy properties for port containers
        def _get_inputs(self):
            if not hasattr(self, '_inputs'):
                self._inputs = InputPortContainer(self.world, self._input_ports, self._input_props)
            return self._inputs

        def _get_outputs(self):
            if not hasattr(self, '_outputs'):
                props = {
                    method.__output_property_name__: method.__get__(self, self.__class__)
                    for method in self.__class__.__dict__.values()
                    if hasattr(method, '__is_output_property__')
                }
                self._outputs = OutputPortContainer(self.world, self._output_ports, props)
            return self._outputs

        # Replace properties with lazy versions
        cls.ins = property(_get_inputs)
        cls.outs = property(_get_outputs)

        return cls
    return decorator


def output_property(name: str):
    """
    Function decorator for defining output properties on a control system.
    """
    def decorator(method):
        method.__is_output_property__ = True
        method.__output_property_name__ = name
        return method
    return decorator


class ControlSystem(ABC):
    """
    Abstract base class for a system with input and output ports. All control systems
    must be decorated with @control_system.
    """
    __slots__ = ["_inputs", "_outputs", "world", "_input_ports", "_output_ports"]

    ins: InputPortContainer
    outs: OutputPortContainer

    def __init__(self, world):
        self.world = world
        world.add_system(self)

    @property
    def should_stop(self):
        return self.world.should_stop

    @abstractmethod
    def run(self):
        """
        Abstract control loop function, to be run in a separate thread.
        Must stop when world.should_stop is True.
        """
        pass
