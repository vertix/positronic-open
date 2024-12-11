import abc
from typing import Dict, Optional, Sequence, Tuple

import mujoco
import numpy as np
from dm_control import mujoco as dm_mujoco
from dm_control.utils import inverse_kinematics as ik

from ironic.utils import FPSCounter
from geom import Transform3D


def xmat_to_quat(xmat):
    site_quat = np.empty(4)
    mujoco.mju_mat2Quat(site_quat, xmat)
    return site_quat


class MetricCalculator(abc.ABC):
    def __init__(self, grace_time: Optional[float] = 0.2):
        self.grace_time = grace_time

    @abc.abstractmethod
    def initialize(self, model: mujoco.MjModel, data: mujoco.MjData):
        pass

    @abc.abstractmethod
    def _update(self, model: mujoco.MjModel, data: mujoco.MjData) -> Dict[str, float]:
        pass

    def update(self, model: mujoco.MjModel, data: mujoco.MjData):
        if self.grace_time is not None and data.time < self.grace_time:
            self.initialize(model, data)
            return
        self._update(model, data)

    @abc.abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class CompositeMetricCalculator(MetricCalculator):
    def __init__(self, metric_calculators: Sequence[MetricCalculator]):
        super().__init__()
        self.metric_calculators = metric_calculators

    def initialize(self, model: mujoco.MjModel, data: mujoco.MjData):
        for calculator in self.metric_calculators:
            calculator.initialize(model, data)

    def _update(self, model: mujoco.MjModel, data: mujoco.MjData):
        for calculator in self.metric_calculators:
            calculator.update(model, data)

    def get_metrics(self) -> Dict[str, float]:
        metrics = {}
        for calculator in self.metric_calculators:
            metrics.update(calculator.get_metrics())
        return metrics

    def reset(self):
        for calculator in self.metric_calculators:
            calculator.reset()


class InverseKinematics:
    def __init__(self, data: mujoco.MjData):
        super().__init__()
        self.joints = [f'joint{i}' for i in range(1, 8)]
        self.physics = dm_mujoco.Physics.from_model(data)

    def recalculate_ik(self, target_robot_position: Transform3D) -> Optional[np.ndarray]:
        """
        Returns None if the IK calculation failed
        """
        result = ik.qpos_from_site_pose(
            physics=self.physics,
            site_name='end_effector',
            target_pos=target_robot_position.translation,
            target_quat=target_robot_position.quaternion,
            joint_names=self.joints,
            rot_weight=0.5,
        )

        if result.success:
            return result.qpos[:7]
        print(f"Failed to calculate IK for {target_robot_position}")
        return None


class MujocoSimulator:
    def __init__(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            simulation_rate: float = 1 / 500,
    ):
        super().__init__()
        self.model = model
        self.data = data
        self.model.opt.timestep = simulation_rate
        self.simulation_fps_counter = FPSCounter('Simulation')
        self.pending_actions = []
        self._initial_position = None

    @property
    def robot_position(self):
        return Transform3D(
            translation=self.data.site('end_effector').xpos.copy(),
            quaternion=xmat_to_quat(self.data.site('end_effector').xmat.copy())
        )

    @property
    def grip(self):
        return self.data.actuator('actuator8').ctrl

    @property
    def joints(self):
        return np.array([self.data.qpos[i] for i in range(7)])

    @property
    def actuator_values(self):
        return self.data.ctrl.copy()

    @property
    def ts_sec(self):
        return self.data.time

    @property
    def ts_ns(self):
        return int(self.ts_sec * 1e9)

    @property
    def ext_force_ee(self):
        return np.zeros(6)  # TODO: implement

    @property
    def ext_force_base(self):
        return np.zeros(6)  # TODO: implement

    @property
    def initial_position(self):
        return self._initial_position

    def step(self):
        mujoco.mj_step(self.model, self.data)
        self.simulation_fps_counter.tick()

    def reset(self, keyframe: str = "home"):
        """
        Reset the simulator to the given keyframe.
        """
        mujoco.mj_resetData(self.model, self.data)
        frame = self.model.keyframe(keyframe)
        self.data.qpos = frame.qpos
        self.data.ctrl = frame.ctrl
        mujoco.mj_forward(self.model, self.data)
        self._initial_position = self.robot_position

    def set_actuator_values(self, actuator_values: np.ndarray):
        for i in range(7):
            self.data.actuator(f'actuator{i + 1}').ctrl = actuator_values[i]

    def set_grip(self, grip: float):
        self.data.actuator('actuator8').ctrl = grip


class MujocoRenderer:
    def __init__(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            camera_names: Sequence[str],
            render_resolution: Tuple[int, int] = (320, 240),
    ):
        super().__init__()
        self.model = model
        self.data = data
        self.renderer = None
        self.render_resolution = render_resolution
        self.observation_fps_counter = FPSCounter('Renderer')
        self.camera_names = camera_names

    def render_frames(self):
        views = {}
        # TODO: make cameras configurable
        for cam_name in self.camera_names:
            self.renderer.update_scene(self.data, camera=cam_name)
            views[cam_name] = self.renderer.render()

        self.observation_fps_counter.tick()
        return views

    def initialize(self):
        """
        Initialize the renderer. This must be called before calling render().
        """
        # in case we have other code which works with OpenGL, we need to initialize the renderer in a separate thread to avoid conflicts
        self.renderer = mujoco.Renderer(self.model, height=self.render_resolution[1], width=self.render_resolution[0])

    def render(self) -> Dict[str, np.ndarray]:
        assert self.renderer is not None, "You must call initialize() before calling render()"
        images = self.render_frames()

        return images

    def close(self):
        self.renderer.close()
