from dataclasses import dataclass
from typing import Tuple, Optional
from threading import Lock
import time

import mujoco
import numpy as np
from dm_control import mujoco as dm_mujoco
from dm_control.utils import inverse_kinematics as ik

from control import ControlSystem, control_system
from control.system import output_property
from control.utils import FPSCounter
from control.world import World
from geom import Transform3D

mjc_lock = Lock()

@dataclass
class DesiredAction:
    position: np.ndarray
    orientation: np.ndarray
    grip: float


def xmat_to_quat(xmat):
    site_quat = np.empty(4)
    mujoco.mju_mat2Quat(site_quat, xmat)
    return site_quat


@control_system(
    inputs=["target_robot_position"],
    outputs=["actuator_values"]
)
class InverseKinematics(ControlSystem):
    def __init__(self, world: "World", data: mujoco.MjData):
        super().__init__(world)
        self.joints = [f'joint{i}' for i in range(1, 8)]
        self.physics = dm_mujoco.Physics.from_model(data)

    def recalculate_ik(self, target_robot_position: Transform3D) -> Optional[np.ndarray]:
        """
        Returns None if the IK calculation failed
        """
        with mjc_lock:
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

    def run(self):
        for _ts, value in self.ins.target_robot_position.read_until_stop():
            self.outs.actuator_values.write(self.recalculate_ik(value), self.world.now_ts)


@control_system(inputs=["actuator_values", "target_grip", "target_position"],
                output_props=["robot_position", "robot_translation", "robot_quaternion", "grip", "joints", "ext_force_ee", "ext_force_base"])
class MujocoSimulator(ControlSystem):
    def __init__(
            self,
            world: World,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            simulation_rate: float = 1 / 60,
    ):
        super().__init__(world)
        self.model = model
        self.data = data
        self.simulation_rate = simulation_rate * 1000
        self.last_simulation_time = None

        self.simulation_fps_counter = FPSCounter('Simulation')

    @output_property('robot_translation')
    def robot_translation(self):
        return self.data.site('end_effector').xpos.copy()

    @output_property('robot_quaternion')
    def robot_quaternion(self):
        return xmat_to_quat(self.data.site('end_effector').xmat)

    @output_property('robot_position')
    def robot_position(self):
        return Transform3D(
            translation=self.data.site('end_effector').xpos.copy(),
            quaternion=xmat_to_quat(self.data.site('end_effector').xmat.copy())
        )

    @output_property('grip')
    def grip(self):
        return self.data.actuator('actuator8').ctrl

    @output_property('joints')
    def joints(self):
        return np.array([self.data.qpos[i] for i in range(7)])

    @output_property('ext_force_ee')
    def ext_force_ee(self):
        return np.zeros(6)

    @output_property('ext_force_base')
    def ext_force_base(self):
        return np.zeros(6)

    def simulate(self):
        with mjc_lock:
            self.last_simulation_time = self.world.now_ts
            mujoco.mj_step(self.model, self.data)
            self.simulation_fps_counter.tick()

    def _init_position(self):
        # Load initial position from keyframe in XML
        frame = self.model.keyframe("home")
        self.data.qpos = frame.qpos
        self.data.ctrl = frame.ctrl

        #mujoco.mj_resetData(self.model, self.data)
        self.simulate()

    def run(self):
        self._init_position()
        while not self.world.should_stop:
            result = self.ins.actuator_values.read_nowait()
            if result is not None:
                _ts, values = result
                if values is not None:
                    for i in range(7):
                        self.data.actuator(f'actuator{i + 1}').ctrl = values[i]

            grip = self.ins.target_grip.read_nowait()
            if grip is not None:
                _ts, grip = grip
                self.data.actuator('actuator8').ctrl = grip

            if self.world.now_ts - self.last_simulation_time >= self.simulation_rate:
                self.simulate()

                grip = self.ins.target_grip.read_nowait()
                if grip is not None:
                    _ts, grip = grip
                    self.data.actuator('actuator8').ctrl = grip
            else:
                time_to_sleep = (self.last_simulation_time + self.simulation_rate - self.world.now_ts) / 1000
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)


@control_system(outputs=["images"])
class MujocoRenderer(ControlSystem):
    def __init__(self, world: World, model, data, render_resolution: Tuple[int, int] = (320, 240), max_fps=60):
        super().__init__(world)
        self.model = model
        self.data = data
        self.renderer = None
        self.render_resolution = render_resolution
        self.max_fps = max_fps
        self.last_render_time = -1
        self.observation_fps_counter = FPSCounter('Observation')

    def render_frames(self):
        self.last_render_time = self.world.now_ts

        views = {}

        # TODO: make cameras configurable
        for cam_name in ['top', 'side', 'handcam_left', 'handcam_right']:
            with mjc_lock:
                self.renderer.update_scene(self.data, camera=cam_name)
                views[cam_name] = self.renderer.render()

        self.observation_fps_counter.tick()
        return views

    def run(self):
        self.renderer = mujoco.Renderer(self.model, height=self.render_resolution[1], width=self.render_resolution[0])
        while not self.world.should_stop:
            if self.world.now_ts - self.last_render_time >= 1 / self.max_fps:
                images = self.render_frames()
                self.outs.images.write(images, self.world.now_ts)
            else:
                time_to_sleep = (self.last_render_time + 1 / self.max_fps - self.world.now_ts)
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)

        self.renderer.close()

