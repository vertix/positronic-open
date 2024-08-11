from abc import ABC, abstractmethod
import asyncio
from typing import Any, List


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
        self.bound_to = None
        self.queue = asyncio.Queue()

    def bind(self, port: OutputPort):
        """
        Bind this input port to an output port.
        """
        if self.bound_to is not None:
            self.bound_to.bound_to.remove(self)
        self.bound_to = port
        port.bound_to.append(self)

    def write(self, value: Any):
        """
        Send a value to the parent system's control loop.
        """
        self.queue.put_nowait((self.name, value))


class PortContainer:
    def __init__(self, ports: List[str]):
        self._ports = {name: OutputPort() for name in ports}

    def __getattr__(self, name: str):
        return self._ports[name]

    def __getitem__(self, name: str):
        return self._ports[name]


class InputPortContainer(PortContainer):
    def __init__(self, ports: List[str]):
        self._ports = {name: InputPort(self, name) for name in ports}

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

    async def read(self):
        """
        Async generator to yield values from any of the input ports as they arrive.
        """
        tasks = {asyncio.create_task(port.queue.get()): port.name
                 for port in self._ports.values()}

        while tasks:
            done, pending = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                name = tasks.pop(task)
                value = await task
                yield name, value
                tasks[asyncio.create_task(self._ports[name].queue.get())] = name


class ControlSystem(ABC):
    """
    Abstract base class for a system with input and output ports.
    """
    def __init__(self, *, inputs: List[str] = [], outputs: List[str] = []):
        self._inputs = InputPortContainer(inputs)
        self._outputs = PortContainer(outputs)

        self._control_loop = self.run()

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
    async def run(self):
        """
        Abstract control loop coroutine.
        """
        pass


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

    async def run(self):
        """
        Control loop coroutine that handles events.
        """
        self.on_start()
        async for name, value in self._inputs.read():
            if name in self._handlers:
                self._handlers[name](value)
            self.on_after_input()
        self.on_stop()
