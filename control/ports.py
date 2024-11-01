from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import queue
import time
from typing import Any, Callable, Dict, List, Optional
import threading


class OutputPort:
    """
    Represents an output port that can write values to bound input ports.
    """
    def __init__(self, world):
        self._bound_to = []
        self._world = world

    def write(self, value: Any, timestamp: Optional[int] = None):
        """
        Write a value to all bound input ports.
        """
        for port in self._bound_to:
            port(value, timestamp)

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


# TODO: Let users to subscribe to a port with a callback.
class InputPort(ABC):
    @abstractmethod
    def read(self, block: bool = True, timeout: Optional[float] = None):
        """
        Returns (timestamp, value) or None if timeout is reached.
        Blocking if no timeout. Implementations must respect world.should_stop.
        """
        pass

    def read_nowait(self, timeout: Optional[float] = None):
        return self.read(block=False, timeout=timeout)

    def read_until_stop(self):
        """
        Generator that continuously reads from the input port until the world should stop,
        or until the port is closed.

        This method blocks and yields (timestamp, value) tuples from the input port.
        It stops when the world's should_stop flag is set.
        """
        while not self.world.should_stop:
            res = self.read(block=True, timeout=None)
            if res is None:
                break
            yield res

    @property
    def last(self):
        """
        Returns the last value read from the input port.

        This method will consume all available values in the queue to ensure that the most recent value is returned.
        It will return a tuple of (timestamp, value), raise Empty if the queue is empty, and PortClosed
        if the port is closed.
        """
        return None


class DirectWriteInputPort(InputPort):
    def __init__(self, world):
        self.world = world
        self.queue = deque()
        self._last_value = None
        self._last_value_lock = threading.Lock()


    def write(self, value: Any, timestamp: Optional[int] = None):
        self.queue.append((timestamp, value))

    def read(self, block: bool = True, timeout: Optional[float] = None):
        start_time = time.time()
        while not self.world.should_stop:
            if self.queue:
                with self._last_value_lock:
                    self._last_value = self.queue.popleft()
                return self._last_value
            if not block:
                return None
            if timeout is not None and (time.time() - start_time) >= timeout:
                return None
            time.sleep(0.01)
        return None

    @property
    def last(self):
        while self.read_nowait() is not None:
            pass
        with self._last_value_lock:
            return self._last_value


class ThreadedInputPort(InputPort):
    def __init__(self, world, binded_to: OutputPort):
        self.queue = queue.Queue(maxsize=5)
        binded_to._bind(self._write)
        self.world = world

        self._last_value = None
        self._last_value_lock = threading.Lock()

    def _write(self, value: Any, timestamp: Optional[int] = None):
        self.queue.put((timestamp, value))

    def read(self, block: bool = True, timeout: Optional[float] = None):
        TICK = 1
        while not self.world.stop_event.is_set():
            try:
                t_o = min(TICK, timeout) if timeout is not None else TICK
                timeout = max(timeout - TICK, 0) if timeout is not None else None
                result = self.queue.get(block=block, timeout=t_o)
                self.queue.task_done()
                with self._last_value_lock:
                    self._last_value = result
                return result
            except queue.Empty:
                if not block or (timeout is not None and timeout <= 0):
                    return None
        return None  # Stop is requested

    @property
    def last(self):
        while self.read_nowait() is not None:
            pass
        with self._last_value_lock:
            return self._last_value


class OutputPortContainer:
    def __init__(self, world, ports: List[str], props: Dict[str, Callable]):
        self._ports = {name: OutputPort(world) for name in ports}
        self._ports.update(props)

    def __getattr__(self, name: str):
        return self.__getitem__(name)

    def __getitem__(self, name: str):
        return self._ports[name]


class InputPortContainer:
    __slots__ = ["_world", "_ports", "_props"]

    def __init__(self, world, ports: List[str], props: List[str]):
        self._world = world
        self._ports = {name: None for name in ports}
        self._props = {name: None for name in props}

    def _create_port(self, output_port: OutputPort):
        return ThreadedInputPort(self._world, output_port)

    def __getattr__(self, name: str):
        if name in self._ports:
            if self._ports[name] is None:
                self._ports[name] = DirectWriteInputPort(self._world)
            return self._ports[name]
        if name in self._props:
            return self._props[name]
        raise AttributeError(f"Input {name} not found")

    def __getitem__(self, name: str):
        return self.__getattr__(name)

    def __setitem__(self, name: str, output_port: Any):
        """
        Set an attribute and bind input ports / properties if applicable.
        """
        if name in self.__slots__:
            super().__setattr__(name, output_port)
            return

        if name in self._ports:
            if not isinstance(output_port, OutputPort):
                raise TypeError(f"Expected OutputPort, got {type(output_port).__name__}")
            if self._ports[name] is None:
                self._ports[name] = self._create_port(output_port)
            else:
                raise ValueError(f"Port {name} already assigned")
        elif name in self._props:
            if self._props[name] is not None:
                raise ValueError(f"Property {name} already assigned")
            if not callable(output_port):
                raise TypeError(f"Expected callable, got {type(output_port).__name__}")
            self._props[name] = output_port
        else:
            raise ValueError(f"Input {name} not found")

    def __setattr__(self, name: str, output_port: Any):
        self.__setitem__(name, output_port)

    def read(self, timeout: Optional[float] = None):
        """
        Generator to yield values from any of the input ports as they arrive,
        or yield None if the timeout is reached.
        """
        with ThreadPoolExecutor(max_workers=len(self._ports)) as executor:
            futures = {executor.submit(port.read): name
                    for name, port in self._ports.items() if port is not None}
            TICK = 1
            timeout_left = timeout

            while futures and not self._world.stop_event.is_set():
                t_o = min(TICK, timeout_left) if timeout_left is not None else TICK
                timeout_left = max(timeout_left - TICK, 0) if timeout_left is not None else None
                done, _ = wait(futures, timeout=t_o, return_when=FIRST_COMPLETED)

                if not done and (timeout_left is not None and timeout_left <= 0):
                    yield None, None, None
                    timeout_left = timeout
                    continue

                for future in done:
                    name = futures.pop(future)
                    result = future.result()
                    if result is None: return  # Stop is requested
                    timestamp, value = result
                    yield name, timestamp, value

                    futures[executor.submit(self._ports[name].read)] = name
                    timeout_left = timeout

