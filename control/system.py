from abc import ABC, abstractmethod
from typing import Any, Dict, List


class OutputPort:
    """
    Represents an output port that can write values to bound input ports.
    """
    def __init__(self):
        self.bound_to = []

    def write(self, value: Any):
        """
        Write a value to all bound input ports.
        """
        for port in self.bound_to:
            port.write(value)


class InputPort:
    """
    Represents an input port that can bind to an output port and send values to the system's control loop.
    """
    def __init__(self, system: 'ControlSystem', name: str):
        self.system = system
        self.name = name

    def bind(self, port: OutputPort):
        """
        Bind this input port to an output port.
        """
        port.bound_to.append(self)

    def write(self, value: Any):
        """
        Send a value to the system's control loop.
        """
        self.system._control_loop.send((self.name, value))


class PortContainer:
    """
    Container for managing multiple ports.
    """
    def __init__(self, ports: Dict[str, Any]):
        self._ports = ports

    def __getattr__(self, name: str):
        """
        Get a port by name.
        """
        return self._ports[name]

    def __setattr__(self, name: str, value: Any):
        """
        Set an attribute and bind input ports if applicable.
        """
        if name == "_ports":
            self.__dict__[name] = value
            return

        if name not in self._ports:
            raise ValueError(f"Port {name} not found")
        if not isinstance(self._ports[name], InputPort):
            raise ValueError(f"Port {name} is not an InputPort")
        if not isinstance(value, OutputPort):
            raise TypeError(f"Expected OutputPort, got {type(value).__name__}")
        self._ports[name].bind(value)


    def __getitem__(self, name: str):
        """
        Get a port by name using item access.
        """
        return self._ports[name]


class ControlSystem(ABC):
    """
    Abstract base class for a system with input and output ports.
    """
    def __init__(self, *, inputs: List[str] = [], outputs: List[str] = []):
        self._inputs = PortContainer({name: InputPort(self, name) for name in inputs})
        self._outputs = PortContainer({name: OutputPort() for name in outputs})

        self._control_loop = self._control()

    @property
    def ins(self) -> PortContainer:
        """
        Get the input ports.
        """
        return self._inputs

    @property
    def outs(self) -> PortContainer:
        """
        Get the output ports.
        """
        return self._outputs

    @abstractmethod
    def _control(self):
        """
        Abstract control loop coroutine.
        """
        pass

    def start(self):
        """
        Start the control loop.
        """
        self._control_loop.send(None)

    def stop(self):
        """
        Stop the control loop.
        """
        self._control_loop.close()


class EventSystem(ControlSystem):
    """
    System that handles events with registered handlers.
    """
    def __init__(self, *, inputs: List[str] = [], outputs: List[str] = []):
        super().__init__(inputs=inputs, outputs=outputs)
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

    def _control(self):
        """
        Control loop coroutine that handles events.
        """
        self.on_start()
        try:
            while True:
                name, value = yield
                if name in self._handlers:
                    self._handlers[name](value)
                else:
                    raise ValueError(f"Input {name} is not recognized")
                self.on_after_input()
        except GeneratorExit:
            self.on_stop()
