from dataclasses import dataclass
from typing import Tuple, Optional
from threading import Lock

import mujoco
import numpy as np
from dm_control import mujoco as dm_mujoco
from dm_control.utils import inverse_kinematics as ik

from control import ControlSystem, control_system
from control.system import output_property
from control.utils import control_system_fn, FPSCounter
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
    desired_action: Optional[DesiredAction]

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


@control_system(inputs=["actuator_values", "target_grip"],
                outputs=["images"],
                output_props=["robot_position", "grip", "joints", "ext_force_ee", "ext_force_base"])
class Mujoco(ControlSystem):
    def __init__(
            self,
            world: World,
            model,
            data,
            render_resolution: Tuple[int, int] = (320, 240),
            simulation_rate: float = 1 / 60,
            observation_rate: float = 1 / 60
    ):
        super().__init__(world)
        self.model = model
        self.data = data
        self.renderer = None
        self.render_resolution = render_resolution
        self.simulation_rate = simulation_rate * 1000
        self.observation_rate = observation_rate * 1000
        self.last_observation_time = -1
        self.last_simulation_time = None

        self.simulation_fps_counter = FPSCounter('Simulation')
        self.observation_fps_counter = FPSCounter('Observation')

    def render_frames(self):
        views = {}
        with mjc_lock:
            mujoco.mj_forward(self.model, self.data)

        # TODO: make cameras configurable
        for cam_name in ['top', 'side', 'handcam_left', 'handcam_right']:
            self.renderer.update_scene(self.data, camera=cam_name)
            views[cam_name] = self.renderer.render()
        return views

    @output_property('robot_position')
    def robot_position(self):
        return (Transform3D(self.data.site('end_effector').xpos,
                            xmat_to_quat(self.data.site('end_effector').xmat)),
                self.world.now_ts)

    @output_property('grip')
    def grip(self):
        return self.data.actuator('actuator8').ctrl, self.world.now_ts

    @output_property('joints')
    def joints(self):
        return np.array([self.data.qpos[i] for i in range(7)]), self.world.now_ts

    @output_property('ext_force_ee')
    def ext_force_ee(self):
        return np.zeros(6), self.world.now_ts

    @output_property('ext_force_base')
    def ext_force_base(self):
        return np.zeros(6), self.world.now_ts

    def simulate(self):
        with mjc_lock:
            self.last_simulation_time = self.world.now_ts
            mujoco.mj_step(self.model, self.data)
            self.simulation_fps_counter.tick()

    def _init_position(self):
        # TODO: hacky way to set initial position, figure out how to do it via xml
        values = [0, 0.3, 0, -1.57079, 0, 1.92, 0.927, 0.04]

        for i in range(7):
            self.data.actuator(f'actuator{i + 1}').ctrl = values[i]

        for i in range(10):
            self.simulate()


    def run(self):
        self.renderer = mujoco.Renderer(self.model, height=self.render_resolution[1], width=self.render_resolution[0])
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

            if self.world.now_ts - self.last_observation_time >= self.observation_rate:
                images = self.render_frames()
                self.observation_fps_counter.tick()
                self.outs.images.write(images, self.world.now_ts)

        self.renderer.close()
