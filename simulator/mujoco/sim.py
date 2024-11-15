from typing import Dict, Tuple
import mujoco
import numpy as np

from control.utils import FPSCounter
from geom import Transform3D


def xmat_to_quat(xmat):
    site_quat = np.empty(4)
    mujoco.mju_mat2Quat(site_quat, xmat)
    return site_quat


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
    def ts(self):
        return self.data.time
    
    @property
    def ext_force_ee(self):
        return np.zeros(6)  # TODO: implement

    @property
    def ext_force_base(self):
        return np.zeros(6)  # TODO: implement

    def step(self):
        mujoco.mj_step(self.model, self.data)
        self.simulation_fps_counter.tick()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        frame = self.model.keyframe("home")
        self.data.qpos = frame.qpos
        self.data.ctrl = frame.ctrl


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
            render_resolution: Tuple[int, int] = (320, 240),
    ):
        super().__init__()
        self.model = model
        self.data = data
        self.renderer = None
        self.render_resolution = render_resolution
        self.observation_fps_counter = FPSCounter('Renderer')

    def render_frames(self):
        views = {}

        # mujoco.mj_forward(self.model, self.data)
        # TODO: make cameras configurable
        for cam_name in ['top', 'side', 'handcam_left', 'handcam_right']:
            self.renderer.update_scene(self.data, camera=cam_name)
            views[cam_name] = self.renderer.render()

        self.observation_fps_counter.tick()
        return views
    
    def initialize(self):
        self.renderer = mujoco.Renderer(self.model, height=self.render_resolution[1], width=self.render_resolution[0])

    def render(self) -> Dict[str, np.ndarray]:
        assert self.renderer is not None, "You must call initialize() before calling render()"
        images = self.render_frames()

        return images

    def close(self):
        self.renderer.close()

