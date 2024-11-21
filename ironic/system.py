import asyncio
from dataclasses import dataclass
import time
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, List
import re
import inspect
import warnings


@dataclass
class Message:
    """
    Contains some data and a timestamp for this data. Timestamps are integers,
    to avoid floating point precision issues. It can be related to epoch or
    to anything else, depending on the context.

    If no timestamp is provided, the current system time is used.
    """
    data: Any
    timestamp: int = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = system_clock()


def system_clock() -> int:
    """Get current timestamp in nanoseconds."""
    return time.monotonic_ns()


class OutputPort:
    def __init__(self, name: str):
        self._name = name
        self._handlers = []
        self._logging = False

    def enable_logging(self):
        self._logging = True

    def subscribe(self, handler):
        """Subscribe a handler to this port. Handler must be an async function."""
        self._handlers.append(handler)

    @property
    def name(self):
        return self._name

    async def write(self, message: Message):
        """
        Write a message to the port. Implementations must check that the timestamp
        is in increasing order.
        """
        # We use asyncio.gather and sacrifice reproducibility of results.
        # If you want reproducibility, run handlers sequentially.
        if self._logging:
            print(f"Writing to {self._name}: {message.data}")
        await asyncio.gather(*[handler(message) for handler in self._handlers])

    @property
    def name(self):
        return self._name


def _validate_pythonic_name(name: str, context: str) -> None:
    """
    Validate that a name follows Python naming conventions.
    Allows letters (both cases), numbers and underscores, must start with a letter.
    """
    pattern = re.compile(r'^[a-zA-Z][a-zA-Z0-9_]*$')
    if not pattern.match(name):
        raise ValueError(
            f"Invalid {context} name '{name}'. Names must start with a letter "
            "and contain only letters, numbers, and underscores"
        )


def out_property(method: Callable[..., Awaitable[Message]]) -> Callable[..., Awaitable[Message]]:
    """
    Decorator for declaring that the method is an output property.

    The decorated method must be an async method that returns an `ir.Message`.
    The method name must be listed in output_props of the ironic_system decorator.

    Args:
        method: The async method to decorate

    Returns:
        The decorated method

    Raises:
        ValueError: If the decorated method is not async
    """
    if not inspect.iscoroutinefunction(method):
        raise ValueError(
            f"Output property '{method.__name__}' must be an async method"
        )

    async def wrapper(self, *args, **kwargs) -> Message:
        return await method(self, *args, **kwargs)
    wrapper.__is_output_property__ = True
    wrapper.__output_property_name__ = method.__name__
    return wrapper


def on_message(name: str):
    """
    Decorator for declaring that the method is a callback for a message.

    Args:
        name: The name of the input port this method handles messages from

    The decorated method should have the signature:
        async def method(self, message: Message)

    Raises:
        ValueError: If the decorated method is not async
    """
    def decorator(method):
        if not inspect.iscoroutinefunction(method):
            raise ValueError(
                f"Message handler '{method.__name__}' for port '{name}' must be an async function"
            )

        # Store the port name this method handles
        method.__is_message_handler__ = True
        method.__input_port_name__ = name
        return method
    return decorator


def ironic_system(*, input_ports: List[str] = None, output_ports: List[str] = None,
                   input_props: List[str] = None, output_props: List[str] = None):
    """
    Class decorator for defining ironic system interfaces.
    """
    if input_ports is None: input_ports = []
    if output_ports is None: output_ports = []
    if input_props is None: input_props = []
    if output_props is None: output_props = []

    # Validate naming conventions
    for port in input_ports:
        _validate_pythonic_name(port, "input port")
    for port in output_ports:
        _validate_pythonic_name(port, "output port")
    for prop in input_props:
        _validate_pythonic_name(prop, "input property")
    for prop in output_props:
        _validate_pythonic_name(prop, "output property")

    # Check for duplicate names across all inputs and outputs
    all_inputs = input_ports + input_props
    duplicate_inputs = [x for x in all_inputs if all_inputs.count(x) > 1]
    if duplicate_inputs:
        raise ValueError(f"Duplicate input names declared: {set(duplicate_inputs)}")

    all_outputs = output_ports + output_props
    duplicate_outputs = [x for x in all_outputs if all_outputs.count(x) > 1]
    if duplicate_outputs:
        raise ValueError(f"Duplicate output names declared: {set(duplicate_outputs)}")

    # Warn about input/output name conflicts
    input_output_conflicts = set(all_inputs) & set(all_outputs)
    if input_output_conflicts:
        warnings.warn(
            f"Names used as both input and output: {input_output_conflicts}. "
            "This might make the system harder to understand.",
            UserWarning
        )

    def decorator(cls):
        cls._input_ports = input_ports
        cls._output_ports = output_ports
        cls._input_props = input_props
        cls._output_props = output_props

        # Verify output properties
        defined_props = {
            method.__output_property_name__
            for method in cls.__dict__.values()
            if hasattr(method, '__is_output_property__')
        }

        # Check for properties not declared in output_props
        undeclared_props = defined_props - set(output_props)
        if undeclared_props:
            raise ValueError(
                f"Output properties {undeclared_props} in class {cls.__name__} "
                "are not listed in output_props of @ironic_system"
            )

        # Check for missing properties
        missing_props = set(output_props) - defined_props
        if missing_props:
            raise ValueError(
                f"Output properties {missing_props} are not defined in class {cls.__name__}"
            )

        # Verify message handlers (but don't store them in class)
        handlers = {
            method.__input_port_name__: method
            for method in cls.__dict__.values()
            if hasattr(method, '__is_message_handler__')
        }

        # Check for duplicate handlers
        duplicate_handlers = [x for x in handlers.keys() if list(handlers.keys()).count(x) > 1]
        if duplicate_handlers:
            raise ValueError(
                f"Multiple handlers defined for input ports {set(duplicate_handlers)} "
                f"in class {cls.__name__}"
            )

        return cls
    return decorator


class ControlSystem:
    """A control system that can be connected to other systems through ports and properties.

    A control system has input and output ports for push-based communication (similar to channels),
    and input and output properties for pull-based communication (similar to RPC).

    The system can be either:
    - Passive: only reacts to incoming messages on its input ports
    - Active: also performs periodic work in its step() method

    Ports and properties must be declared using the @ironic_system decorator. Input ports can have
    message handlers defined using @on_message, while output properties are defined using
    @output_property.
    """
    def __init__(self):
        self._message_handlers = {
            method.__input_port_name__: getattr(self, name)  # Get bound methods instead of functions
            for name, method in self.__class__.__dict__.items()
            if hasattr(method, '__is_message_handler__')
        }
        self._output_props = {
            method.__output_property_name__: getattr(self, method.__output_property_name__)
            for method in self.__class__.__dict__.values()
            if hasattr(method, '__is_output_property__')
        }
        outs = {port: OutputPort(port) for port in self._output_ports}
        outs.update(self._output_props)
        self.outs = SimpleNamespace(**outs)
        self.ins = None

    def bind(self, **bindings):
        """Bind inputs to provided outputs of other systems. Must be called before calling setup.

        For convienice, returns self so result can be passed to other functions."""
        if self.ins is None:
            self.ins = SimpleNamespace()

        for name, incoming_output in bindings.items():
            if name in self._input_ports:
                if name in self._message_handlers:  # Otherwise we just ignore that input
                    setattr(self.ins, name, incoming_output.subscribe(self._message_handlers[name]))
            elif name in self._input_props:
                setattr(self.ins, name, incoming_output)  # Property is just a callback, so assignment is enough
            else:
                raise ValueError(f"Unknown input: {name}")
        return self

    async def setup(self):
        """Setup the control system."""
        pass

    async def cleanup(self):
        """Cleanup the control system."""
        pass

    async def step(self):
        """
        Perform periodic work. This method is called repeatedly while the system is running.

        By default does nothing. Override this method to implement active behavior.

        IMPORTANT: If you run your system with other systems (which is what you are here for),
        the step must be non-blocking and return back as soon as possible. If you really need
        to perform blocking operations, do them in a separate thread or asyncio.run_coroutine_threadsafe.
        """
        pass
