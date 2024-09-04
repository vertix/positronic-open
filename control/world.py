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

    async def run(self):
        await asyncio.gather(*[s.run() for s in self._systems])
