from abc import ABC, abstractmethod
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


from control.ports import InputPortContainer, OutputPortContainer


def control_system(*, inputs: List[str] = None, outputs: List[str] = None):
    """
    Class decorator for defining ports on a control system.
    """
    if inputs is None:
        inputs = []
    if outputs is None:
        outputs = []

    def decorator(cls):
        # Store port definitions on class
        cls._input_ports = inputs
        cls._output_ports = outputs

        # Add lazy properties for port containers
        def _get_inputs(self):
            if not hasattr(self, '_inputs'):
                self._inputs = InputPortContainer(self.world, self._input_ports)
            return self._inputs

        def _get_outputs(self):
            if not hasattr(self, '_outputs'):
                self._outputs = OutputPortContainer(self.world, self._output_ports)
            return self._outputs

        # Replace properties with lazy versions
        cls.ins = property(_get_inputs)
        cls.outs = property(_get_outputs)

        return cls
    return decorator


class ControlSystem(ABC):
    """
    Abstract base class for a system with input and output ports. All control systems
    must be decorated with @control_system.
    """
    __slots__ = ["_inputs", "_outputs", "world", "_input_ports", "_output_ports"]
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


class EventSystem(ControlSystem):
    """
    System that handles events with registered handlers. All event systems must be
    decorated with @event_system.
    """
    def __init__(self, world):
        super().__init__(world)
        self._handlers = {}
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, '_event_name'):
                self._handlers[attr._event_name] = attr

    @classmethod
    def on_event(cls, event_name: str):
        """
        Decorator to register an event handler. Example:

        @EventSystem.on_event('joints')
        def on_joints(value):
            self.robot.set_joints(value)
        """
        def decorator(func):
            func._event_name = event_name
            return func
        return decorator

    def on_start(self):
        """
        Hook method called when the system starts.
        """
        pass

    def on_stop(self):
        """
        Hook method called when the system stops.
        """
        pass

    def on_after_input(self):
        """
        Hook method called on any input. It is called after on_event callbacks.
        """
        pass

    def run(self):
        """
        Control loop coroutine that handles events.
        """
        self.on_start()
        read_iter = self.ins.read()
        try:
            while not self.should_stop:
                try:
                    name, ts, value = next(read_iter)
                    if name in self._handlers:
                        self._handlers[name](ts, value)
                    self.on_after_input()
                except StopIteration:
                    return
        finally:
            self.on_stop()
