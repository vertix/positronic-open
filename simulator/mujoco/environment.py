from dataclasses import dataclass
from typing import Tuple, Optional
from threading import Lock
import time

import mujoco
import numpy as np
from dm_control import mujoco as dm_mujoco
from dm_control.utils import inverse_kinematics as ik

from control import ControlSystem, control_system
from control.system import output_property, output_property_custom_time
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
            self.outs.actuator_values.write(self.recalculate_ik(value), _ts)


@control_system(inputs=["actuator_values", "target_grip", "target_position"],
                output_props=[
                    "robot_position",
                    "robot_translation",
                    "robot_quaternion",
                    "grip",
                    "joints",
                    "ext_force_ee",
                    "ext_force_base",
                    "actuator_values",
                    "ts"
                ])
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
        self.model.opt.timestep = simulation_rate
        self.simulation_fps_counter = FPSCounter('Simulation')
        self.simulator_ts = 0
        self.pending_actions = []

    @output_property_custom_time('robot_translation')
    def robot_translation(self):
        return self.data.site('end_effector').xpos.copy(), self.simulator_ts

    @output_property_custom_time('robot_quaternion')
    def robot_quaternion(self):
        return xmat_to_quat(self.data.site('end_effector').xmat), self.simulator_ts

    @output_property_custom_time('robot_position')
    def robot_position(self):
        return Transform3D(
            translation=self.data.site('end_effector').xpos.copy(),
            quaternion=xmat_to_quat(self.data.site('end_effector').xmat.copy())
        ), self.simulator_ts    

    @output_property_custom_time('grip')
    def grip(self):
        return self.data.actuator('actuator8').ctrl, self.simulator_ts

    @output_property_custom_time('joints')
    def joints(self):
        return np.array([self.data.qpos[i] for i in range(7)]), self.simulator_ts

    @output_property_custom_time('ext_force_ee')
    def ext_force_ee(self):
        return np.zeros(6), self.simulator_ts

    @output_property_custom_time('ext_force_base')
    def ext_force_base(self):
        return np.zeros(6), self.simulator_ts

    @output_property_custom_time('actuator_values')
    def actuator_values(self):
        return self.data.ctrl.copy(), self.simulator_ts

    @output_property_custom_time('ts')
    def ts(self):
        return self.simulator_ts

    def simulate(self):
        with mjc_lock:
            self.last_simulation_time = self.world.now_ts
            mujoco.mj_step(self.model, self.data)
            self.simulator_ts += self.simulation_rate
            self.simulation_fps_counter.tick()

    def _init_position(self):
        frame = self.model.keyframe("home")
        self.data.qpos = frame.qpos
        self.data.ctrl = frame.ctrl

        self.simulate()

    def _apply_pending_actions(self):
        # Apply any pending actions that should be executed at current simulation time
        remaining_actions = []
        for action_ts, action_type, action_value in self.pending_actions:
            if action_ts <= self.simulator_ts:
                if action_type == 'actuator_values':
                    for i in range(7):
                        self.data.actuator(f'actuator{i + 1}').ctrl = action_value[i]
                elif action_type == 'grip':
                    self.data.actuator('actuator8').ctrl = action_value
            else:
                remaining_actions.append((action_ts, action_type, action_value))
        self.pending_actions = remaining_actions

    def _handle_inputs(self):
        result = self.ins.actuator_values.read_nowait()
        if result is not None:
            ts, values = result
            if values is not None:
                self.pending_actions.append((ts, 'actuator_values', values))

        grip = self.ins.target_grip.read_nowait()
        if grip is not None:
            ts, grip_value = grip
            self.pending_actions.append((ts, 'grip', grip_value))

    def run(self):
        self._init_position()

        while not self.world.should_stop:
            self._handle_inputs()
            self._apply_pending_actions()

            if self.world.now_ts - self.last_simulation_time >= self.simulation_rate:
                # simulate number of steps to catch up to current time
                steps_to_simulate = int((self.world.now_ts - self.last_simulation_time) / self.simulation_rate)
                for _ in range(steps_to_simulate):
                    self.simulate()
            else:
                time_to_sleep = (self.last_simulation_time + self.simulation_rate - self.world.now_ts) / 1000
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)


@control_system(outputs=["images"])
class MujocoRenderer(ControlSystem):
    def __init__(
            self,
            world: World,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            render_resolution: Tuple[int, int] = (320, 240),
            max_fps: int = 60
    ):
        super().__init__(world)
        self.model = model
        self.data = data
        self.renderer = None
        self.render_resolution = render_resolution
        self.render_rate = 1 / max_fps * 1000
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
            if self.world.now_ts - self.last_render_time >= self.render_rate:
                images = self.render_frames()
                self.outs.images.write(images, self.world.now_ts)
            else:
                time_to_sleep = (self.last_render_time + self.render_rate - self.world.now_ts) / 1000
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)

        self.renderer.close()

