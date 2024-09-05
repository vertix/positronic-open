from abc import ABC, abstractmethod
import logging
from typing import Any, List

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


from control.ports import InputPortContainer, OutputPortContainer

class ControlSystem(ABC):
    """
    Abstract base class for a system with input and output ports.
    """
    def __init__(self, world, *, inputs: List[str] = [], outputs: List[str] = []):
        self._inputs = InputPortContainer(world, inputs)
        self._outputs = OutputPortContainer(world, outputs)
        self.world = world
        world.add_system(self)

    @property
    def ins(self) -> InputPortContainer:
        """
        Get the input ports.
        """
        return self._inputs

    @property
    def outs(self) -> OutputPortContainer:
        """
        Get the output ports.
        """
        return self._outputs

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
    System that handles events with registered handlers.
    """
    def __init__(self, world, *, inputs: List[str] = [], outputs: List[str] = []):
        super().__init__(world, inputs=inputs, outputs=outputs)
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
            while not self._world.should_stop:
                try:
                    name, ts, value = next(read_iter)
                    if name in self._handlers:
                        self._handlers[name](ts, value)
                    self.on_after_input()
                except StopIteration:
                    return
        finally:
            self.on_stop()
