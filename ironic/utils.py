import asyncio
import time
import signal
from typing import Callable, Optional

from ironic.system import ControlSystem

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
