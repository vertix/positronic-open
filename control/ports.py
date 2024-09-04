import asyncio
from typing import Any, List, Optional

class OutputPort:
    """
    Represents an output port that can write values to bound input ports.
    """
    def __init__(self, world):
        self.bound_to = []
        self.world = world

    async def write(self, value: Any, timestamp: Optional[int] = None):
        """
        Write a value to all bound input ports.
        """
        for port in self.bound_to:
            await port.write(value, timestamp)

    @property
    def subscribed(self):
        """
        Check if the port is subscribed to by any input ports.
        """
        return len(self.bound_to) > 0


class AsyncioInputPort:
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

    async def write(self, value: Any, timestamp: Optional[int] = None):
        """
        Send a value to the parent system's control loop.
        If there are unread values, they are discarded.
        """
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
        self._ports = {}
        self._port_names = ports

    def _create_port(self, name: str, output_port: OutputPort):
        if self.world == output_port.world:
            res = AsyncioInputPort(self.world, name)
            res.bind(output_port)
            return res
        else:
            raise ValueError("Cross world binding is not supported")

    def __setattr__(self, name: str, output_port: Any):
        """
        Set an attribute and bind input ports if applicable.
        """
        if name in {"world", "_ports", "_port_names"}:
            self.__dict__[name] = output_port
            return

        if name not in self._port_names:
            raise ValueError(f"Port {name} not found")
        # if not isinstance(self._ports[name], InputPort):
        #     raise ValueError(f"Port {name} is not an InputPort")
        if not isinstance(output_port, OutputPort):
            raise TypeError(f"Expected OutputPort, got {type(output_port).__name__}")
        if name not in self._ports:
            self._ports[name] = self._create_port(name, output_port)
        else:
            raise ValueError(f"Port {name} already assigned")

    async def read(self, timeout: Optional[float] = None):
        """
        Async generator to yield values from any of the input ports as they arrive,
        or yield None if the timeout is reached.
        """
        tasks = {asyncio.create_task(port.queue.get()): port.name
                 for port in self._ports.values()}

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
