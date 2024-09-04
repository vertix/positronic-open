from abc import abstractmethod
import asyncio
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
