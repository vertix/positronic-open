from abc import abstractmethod
import asyncio
import threading
from typing import List

from control.system import ControlSystem

# World is a space where ControlSystems live. One world means one control loop
# to rule them all.
class World:
    def __init__(self):
        self._systems: List[ControlSystem] = []

    def add_system(self, system: ControlSystem):
        self._systems.append(system)

    @abstractmethod
    async def run(self):
        pass


class MainThreadWorld(World):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MainThreadWorld, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            super().__init__()
            self._initialized = True

    async def run(self):
        await asyncio.gather(*[s.run() for s in self._systems])


class ThreadWorld(World):
    async def run(self):
        stop_event = threading.Event()

        async def _run_systems():
            try:
                await asyncio.gather(*[s.run() for s in self._systems])
            except asyncio.CancelledError:
                print("Systems cancelled")

        def thread_target():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            task = loop.create_task(_run_systems())

            try:
                while not stop_event.is_set():
                    loop.run_until_complete(asyncio.sleep(0.1))
            finally:
                task.cancel()
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

        thread = threading.Thread(target=thread_target)
        try:
            thread.start()
            thread.join()
        finally:
            print("Stopping ThreadWorld")
            stop_event.set()
            if thread.is_alive():
                thread.join()
            print("ThreadWorld stopped")
