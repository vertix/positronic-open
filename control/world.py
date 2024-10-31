from abc import ABC, abstractmethod
import threading
import time
from typing import List

from control.system import ControlSystem

# World is a space where ControlSystems live. One world means one control loop
# to rule them all.
class World(ABC):
    def __init__(self):
        self._systems: List[ControlSystem] = []
        self.stop_event = threading.Event()

    def add_system(self, system: ControlSystem):
        self._systems.append(system)

    @property
    def should_stop(self):
        return self.stop_event.is_set()

    @property
    @abstractmethod
    def now_ts(self) -> int:
        pass

    @abstractmethod
    def run(self):
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
            self._mono_delta = time.time() - time.monotonic()

    @property
    def now_ts(self) -> int:
        return int((time.monotonic() + self._mono_delta) * 1000)

    def run(self):
        threads = []
        for system in self._systems:
            thread = threading.Thread(
                target=self._run_system,
                args=(system,),
                name=f"{system.__class__.__name__}"
            )
            thread.start()
            threads.append(thread)

        try:
            while not self.stop_event.is_set():
                if not any(thread.is_alive() for thread in threads):
                    break
                time.sleep(0.5)
        finally:
            self.stop_event.set()
            for thread in threads:
                thread.join()

    def _run_system(self, system):
        try:
            system.run()
        except Exception as e:
            print(f"Exception in system {system.__class__.__name__}: {e}")
            import traceback
            traceback.print_exc()
