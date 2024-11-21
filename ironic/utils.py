import asyncio
import time
import signal
from typing import Any, Callable, Optional

from ironic.system import ControlSystem, Message, OutputPort

class FPSCounter:
    """Utility class for tracking and reporting frames per second (FPS).

    Counts frames and periodically reports the average FPS over the reporting interval.

    Args:
        prefix (str): Prefix string to use in FPS report messages
        report_every_sec (float): How often to report FPS, in seconds (default: 10.0)
    """
    def __init__(self, prefix: str, report_every_sec: float = 10.0):
        self.prefix = prefix
        self.report_every_sec = report_every_sec
        self.reset()

    def reset(self):
        self.last_report_time = time.monotonic()
        self.frame_count = 0

    def report(self):
        fps = self.frame_count / (time.monotonic() - self.last_report_time)
        print(f"{self.prefix}: {fps:.2f} fps")
        self.last_report_time = time.monotonic()
        self.frame_count = 0

    def tick(self):
        self.frame_count += 1
        if time.monotonic() - self.last_report_time >= self.report_every_sec:
            self.report()


async def run_gracefully(system: ControlSystem, extra_cleanup_fn: Optional[Callable[[], None]] = None):
    """Runs a control system with graceful shutdown handling.

    This function manages the lifecycle of a ControlSystem, handling setup, continuous operation,
    and cleanup. It sets up signal handlers to catch interrupts (Ctrl+C) and ensures proper
    cleanup of resources when shutting down.

    Args:
        system (ControlSystem): The control system instance to run
        extra_cleanup_fn (Optional[Callable[[], None]]): Optional callback function to perform
            additional cleanup tasks after the system cleanup

    Example:
        ```python
        system = MyControlSystem()
        def cleanup():
            print("Performing extra cleanup...")

        await run_gracefully(system, extra_cleanup_fn=cleanup)
        ```
    """
    shutdown_event = asyncio.Event()
    def signal_handler(signal, frame):
        print("Program interrupted by user, exiting...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        await system.setup()
        while not shutdown_event.is_set():
            await system.step()
    finally:
        await system.cleanup()
        print('System cleanup finished')
        if extra_cleanup_fn:
            extra_cleanup_fn()
            print('Extra cleanup finished')


def map_property(function: Callable[[Any], Any], property: Callable[[], Message]):
    """Creates a new property that transforms the data of another property using a mapping function.

    This utility is useful for connecting systems that expect different data formats. It preserves
    the timestamp of the original message while transforming its data content.

    Args:
        function: A function that transforms the data from one format to another
        property: The source property function that returns Messages

    Returns:
        An async function that returns a Message with transformed data and original timestamp

    Example:
        ```python
        # Connect two systems with different position formats
        robot.bind(
            target_position=map_property(
                lambda pos: transform_coordinates(pos),
                inference.outs.target_robot_position
            )
        )
        ```
    """
    async def result():
        original_message = await property()
        return Message(data=function(original_message.data), timestamp=original_message.timestamp)
    return result


def map_port(function: Callable[[Any], Any], port: OutputPort) -> OutputPort:
    """Creates a new port that transforms the data of another port using a mapping function.

    This utility is useful for connecting systems that expect different data formats. It preserves
    the timestamp of the original message while transforming its data content.
    """
    fn_name = getattr(function, '__name__', 'mapped_port')
    mapped_port = OutputPort(f"{port.name}_{fn_name}")

    async def handler(message: Message):
        transformed_data = function(message.data)
        await mapped_port.write(Message(transformed_data, timestamp=message.timestamp))

    port.subscribe(handler)
    return mapped_port


def properties_dict(**properties):
    """Creates a property that returns a dictionary of multiple property values.

    Args:
        **properties: Keyword arguments mapping property names to property functions
            that return Messages

    Returns:
        An async function that returns a Message containing a dictionary of property values
    """
    async def result():
        # Gather all property values concurrently
        messages = await asyncio.gather(*(prop_fn() for prop_fn in properties.values()))

        # Build the dictionaries
        prop_values = {
            name: messages[i].data
            for i, name in enumerate(properties.keys())
        }
        timestamps = [msg.timestamp for msg in messages]

        # Warn if time range is too large
        if timestamps:
            time_range = (max(timestamps) - min(timestamps)) / 1e6  # Convert ns to ms
            if time_range > 10:
                print(f"Warning: time range for property values is {time_range:.1f} ms")

        return Message(data=prop_values, timestamp=min(timestamps))

    return result


def fps_counter(prefix: str, report_every_sec: float = 10.0):
    """
    A decorator that prints the FPS of the decorated function.

    Args:
        prefix: prefix for the FPS report
        report_every_sec: time in seconds between reports
    Example:
        >>> @fps_counter("render")
        >>> def render(self):
        >>>     pass
    """
    def decorator(fn):
        fps_counter = FPSCounter(prefix, report_every_sec)
        def wrapper(*args, **kwargs):
            fps_counter.tick()
            return fn(*args, **kwargs)
        return wrapper
    return decorator
