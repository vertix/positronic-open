from abc import ABC, abstractmethod
import asyncio
from typing import Any, List, Optional

class OutputPort:
    """
    Represents an output port that can write values to bound input ports.
    """
    def __init__(self, world):
        self._bound_to = []
        self._world = world

    async def write(self, value: Any, timestamp: Optional[int] = None):
        """
        Write a value to all bound input ports.
        """
        for port in self._bound_to:
            await port(value, timestamp)

    def _bind(self, callback):
        self._bound_to.append(callback)

    @property
    def world(self):
        return self._world

    @property
    def subscribed(self):
        """
        Check if the port is subscribed to by any input ports.
        """
        return len(self._bound_to) > 0


class InputPort(ABC):
    @abstractmethod
    async def read(self, timeout: Optional[float] = None):
        "Returns (timestamp, value) or None if timeout is reached"
        pass


class AsyncioInputPort(InputPort):
    """
    Represents an input port that can bind to an output port and send values to the system's control loop.
    """
    def __init__(self, binded_to: OutputPort):
        self.queue = asyncio.Queue(maxsize=5)
        binded_to._bind(self._write)

    async def _write(self, value: Any, timestamp: Optional[int] = None):
        # if self.queue.full():
        #     await self.queue.get()
        #     self.queue.task_done()
        await self.queue.put((timestamp, value))
        await asyncio.sleep(0)  # Yield control to let readers to catch up

    async def read(self, timeout: Optional[float] = None):
        """
        Read a value from the input port. Returns None if timeout is reached.
        """
        try:
            if timeout is None:
                res = await self.queue.get()
            else:
                res = await asyncio.wait_for(self.queue.get(), timeout)
        except asyncio.TimeoutError:
            return None
        self.queue.task_done()
        return res


class OutputPortContainer:
    def __init__(self, world, ports: List[str]):
        self._ports = {name: OutputPort(world) for name in ports}

    def __getattr__(self, name: str):
        return self._ports[name]

    def __getitem__(self, name: str):
        return self._ports[name]


class InputPortContainer(OutputPortContainer):
    def __init__(self, world, ports: List[str]):
        self.world = world
        self._ports = {name: None for name in ports}

    def _create_port(self, name: str, output_port: OutputPort):
        if self.world == output_port.world:
            return AsyncioInputPort(output_port)
        else:
            raise ValueError("Cross world binding is not supported")

    def __setattr__(self, name: str, output_port: Any):
        """
        Set an attribute and bind input ports if applicable.
        """
        if name in {"world", "_ports"}:
            self.__dict__[name] = output_port
            return

        if name not in self._ports:
            raise ValueError(f"Port {name} not found")
        if not isinstance(output_port, OutputPort):
            raise TypeError(f"Expected OutputPort, got {type(output_port).__name__}")
        if self._ports[name] is None:
            self._ports[name] = self._create_port(name, output_port)
        else:
            raise ValueError(f"Port {name} already assigned")

    async def read(self, timeout: Optional[float] = None):
        """
        Async generator to yield values from any of the input ports as they arrive,
        or yield None if the timeout is reached.
        """
        tasks = {asyncio.create_task(port.queue.get()): name
                 for name, port in self._ports.items() if port is not None}

        while tasks:
            done, _ = await asyncio.wait(tasks.keys(), timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                yield None, None, None
                continue

            for task in done:
                name = tasks.pop(task)
                value = task.result()
                yield name, value[0], value[1]
                tasks[asyncio.create_task(self._ports[name].queue.get())] = name
