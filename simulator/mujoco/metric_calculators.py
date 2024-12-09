from typing import Dict
import mujoco
import numpy as np

from simulator.mujoco.sim import MetricCalculator


class ObjectMovedCalculator(MetricCalculator):
    def __init__(self, object_name: str, threshold: float):
        super().__init__()
        self.object_name = object_name
        self.threshold = threshold
        self.moved = False

    def initialize(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.initial_position = data.body(self.object_name).xpos.copy()

    def _update(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        Update the calculator with the current state of the simulator.
        """

        current_position = data.body(self.object_name).xpos
        distance_moved = np.linalg.norm(current_position - self.initial_position)
        self.moved = distance_moved > self.threshold

    def get_metrics(self) -> Dict[str, float]:
        return {'object_moved': float(self.moved)}

    def reset(self):
        self.moved = False


class ObjectDistanceCalculator(MetricCalculator):
    def __init__(self, object_1: str, object_2: str):
        super().__init__()
        self.object_1 = object_1
        self.object_2 = object_2
        self.distance = 0.0

    def initialize(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.initial_position_1 = data.body(self.object_1).xpos.copy()
        self.initial_position_2 = data.body(self.object_2).xpos.copy()

    def _update(self, model: mujoco.MjModel, data: mujoco.MjData):
        current_position_1 = data.body(self.object_1).xpos
        current_position_2 = data.body(self.object_2).xpos
        self.distance = np.linalg.norm(current_position_1 - current_position_2)

    def get_metrics(self) -> Dict[str, float]:
        return {'object_distance': float(self.distance)}

    def reset(self):
        self.distance = 0.0


class ObjectLiftedTimeCalculator(MetricCalculator):
    def __init__(self, object_name: str, threshold: float):
        super().__init__()
        self.object_name = object_name
        self.threshold = threshold
        self.lifted_time = 0.0

    def initialize(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.initial_position = data.body(self.object_name).xpos.copy()

    def _update(self, model: mujoco.MjModel, data: mujoco.MjData):
        current_position = data.body(self.object_name).xpos
        if current_position[2] > self.initial_position[2] + self.threshold:
            self.lifted_time = data.time

    def get_metrics(self) -> Dict[str, float]:
        return {'object_lifted_time': float(self.lifted_time)}

    def reset(self):
        self.lifted_time = 0.0


class ObjectsStackedCalculator(MetricCalculator):
    def __init__(self, object_1: str, object_2: str, velocity_threshold: float):
        super().__init__(grace_time=None)
        self.object_1 = object_1
        self.object_2 = object_2
        self.stacked = False
        self.velocity_threshold = velocity_threshold

    def initialize(self, model: mujoco.MjModel, data: mujoco.MjData):
        pass

    def _update(self, model: mujoco.MjModel, data: mujoco.MjData):
        # assume objects stacked if their velocities are close to 0 and their mass centers are closer then their radius

        vel1 = data.body(self.object_1).subtree_linvel
        vel1 = np.linalg.norm(vel1)
        vel2 = data.body(self.object_2).subtree_linvel
        vel2 = np.linalg.norm(vel2)

        obj1_size = model.geom_size[data.body(self.object_1).id]
        obj1_size = np.linalg.norm(obj1_size)
        obj2_size = model.geom_size[data.body(self.object_2).id]
        obj2_size = np.linalg.norm(obj2_size)

        not_moving = vel1 < self.velocity_threshold and vel2 < self.velocity_threshold

        close_enough = np.linalg.norm(data.body(self.object_1).xpos - data.body(self.object_2).xpos) < (obj1_size + obj2_size)

        self.stacked = not_moving and close_enough

    def get_metrics(self) -> Dict[str, float]:
        return {'objects_stacked': float(self.stacked)}

    def reset(self):
        self.stacked = False
