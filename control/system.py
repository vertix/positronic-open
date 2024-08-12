from abc import ABC, abstractmethod
import asyncio
import logging
from typing import Any, List

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class OutputPort:
    """
    Represents an output port that can write values to bound input ports.
    """
    def __init__(self):
        self.bound_to = []

    async def write(self, value: Any):
        """
        Write a value to all bound input ports.
        """
        for port in self.bound_to:
            await port.write(value)


class InputPort:
    """
    Represents an input port that can bind to an output port and send values to the system's control loop.
    """
    def __init__(self, system: 'ControlSystem', name: str):
        self.system = system
        self.name = name
        self.bound_to = None
        self.queue = asyncio.Queue(maxsize=5)

    def bind(self, port: OutputPort):
        """
        Bind this input port to an output port.
        """
        if self.bound_to is not None:
            self.bound_to.bound_to.remove(self)
        self.bound_to = port
        port.bound_to.append(self)

    async def write(self, value: Any):
        """
        Send a value to the parent system's control loop.
        If there are unread values, they are discarded.
        """
        while not self.queue.empty():
            self.queue.get_nowait()
        logger.debug(f"Write to {self.name}")
        await self.queue.put(value)

    async def read(self, timeout: float = None):
        """
        Read a value from the input port. Returns None if timeout is reached.
        """
        try:
            return await asyncio.wait_for(self.queue.get(), timeout)
        except asyncio.TimeoutError:
            return None


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

    async def read(self, timeout: float = None):
        """
        Async generator to yield values from any of the input ports as they arrive,
        or yield None if the timeout is reached.
        """
        tasks = {asyncio.create_task(port.queue.get()): port.name
                 for port in self._ports.values()}

        while tasks:
            done, _ = await asyncio.wait(tasks.keys(), timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                yield None, None
                continue

            for task in done:
                name = tasks.pop(task)
                value = await task
                logger.debug(f"Read from {name}")
                yield name, value
                tasks[asyncio.create_task(self._ports[name].queue.get())] = name


class ControlSystem(ABC):
    """
    Abstract base class for a system with input and output ports.
    """
    def __init__(self, *, inputs: List[str] = [], outputs: List[str] = []):
        self._inputs = InputPortContainer(inputs)
        self._outputs = PortContainer(outputs)

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

    async def on_start(self):
        """
        Hook method called when the system starts.
        """
        pass

    async def on_stop(self):
        """
        Hook method called when the system stops.
        """
        pass

    async def on_after_input(self):
        """
        Hook method called on any input. It is called after on_event callbacks.
        """
        pass

    async def run(self):
        """
        Control loop coroutine that handles events.
        """
        await self.on_start()
        try:
            async for name, value in self.ins.read():
                if name in self._handlers:
                    await self._handlers[name](value)
                await self.on_after_input()
        except asyncio.CancelledError:
            pass
        finally:
            await self.on_stop()
