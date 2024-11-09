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
from geom import Transform3D

mjc_lock = Lock()

@dataclass
class DesiredAction:
    position: np.ndarray
    orientation: np.ndarray
    grip: float


@dataclass
class ActuatorValues:
    values: np.ndarray
    grip: float
    success: bool


@dataclass
class Observation:
    position: np.ndarray
    orientation: np.ndarray
    grip: float
    joints: np.ndarray

    # Images
    top_image: np.ndarray
    side_image: np.ndarray
    handcam_left_image: np.ndarray
    handcam_right_image: np.ndarray


def xmat_to_quat(xmat):
    site_quat = np.empty(4)
    mujoco.mju_mat2Quat(site_quat, xmat)
    return site_quat


@control_system(
    inputs=["desired_action"],
    outputs=["actuator_values"]
)
class InverseKinematics(ControlSystem):
    desired_action: Optional[DesiredAction]

    def __init__(self, world: "World", data: mujoco.MjData):
        super().__init__(world)
        self.joints = [f'joint{i}' for i in range(1, 8)]

        self.physics = dm_mujoco.Physics.from_model(data)
        self.desired_action = None


    def recalculate_ik(self) -> ActuatorValues:
        if self.desired_action is None:
            return ActuatorValues(values=np.zeros(7), grip=0.0, success=False)

        with mjc_lock:
            result = ik.qpos_from_site_pose(
                physics=self.physics,
                site_name='end_effector',
                target_pos=self.desired_action.position,
                target_quat=self.desired_action.orientation,
                joint_names=self.joints,
                rot_weight=0.5,
            )

        if result.success:
            return ActuatorValues(values=result.qpos[:7], grip=self.desired_action.grip, success=True)

        print(f"Failed to calculate IK for {self.desired_action.position}")
        return ActuatorValues(values=np.zeros(7), grip=self.desired_action.grip, success=False)

    def run(self):
        for _ts, value in self.ins.desired_action.read_until_stop():
            self.desired_action = value
            values = self.recalculate_ik()
            self.outs.actuator_values.write(values, self.world.now_ts)


@control_system_fn(
    inputs=["observation", "desired_action"],
    outputs=['image',
           'ext_force_ee', 'ext_force_base', 'robot_position', 'robot_joints', 'grip',
           'start_episode', 'end_episode',
           'target_grip', 'target_robot_position']
)
def extract_information_to_dump(ins, outs):
    for name, ts, value in ins.read():
        if name == 'observation':
            obs = value
            image = np.hstack([obs.handcam_left_image, obs.handcam_right_image])
            outs.image.write(image, ts)
            outs.robot_position.write(Transform3D(obs.position, obs.orientation), ts)
            outs.grip.write(obs.grip, ts)

            # TODO: add external forces
            outs.ext_force_ee.write(np.zeros(6), ts)
            outs.ext_force_base.write(np.zeros(6), ts)

            outs.robot_joints.write(obs.joints, ts)

        if name == 'desired_action':
            outs.target_grip.write(value.grip, ts)
            outs.target_robot_position.write(Transform3D(value.position, value.orientation), ts)


@control_system(inputs=["actuator_values"],
                outputs=["observation"],
                output_props=["robot_position"])
class Mujoco(ControlSystem):
    def __init__(
            self,
            world: "World",
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

    def get_observation(self):
        self.last_observation_time = self.world.now_ts

        data = {'sensor': self.data.sensordata}
        images = self.render_frames()

        for cam_name, image in images.items():
            data[cam_name] = image

        self.observation_fps_counter.tick()

        return data

    @output_property('robot_position')
    def robot_position(self):
        return (Transform3D(self.data.site('end_effector').xpos,
                            xmat_to_quat(self.data.site('end_effector').xmat)),
                self.world.now_ts)

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

        self.last_fps_print_time = self.world.now_ts

        while not self.world.should_stop:
            result = self.ins.actuator_values.read_nowait()
            if result is not None:
                _ts, value = result
                if value.success:
                    for i in range(7):
                        self.data.actuator(f'actuator{i + 1}').ctrl = value.values[i]
                    self.data.actuator('actuator8').ctrl = value.grip

            if self.world.now_ts - self.last_simulation_time >= self.simulation_rate:
                self.simulate()

            if self.world.now_ts - self.last_observation_time >= self.observation_rate:
                obs = self.get_observation()

                observation = Observation(
                    position=self.data.site('end_effector').xpos,
                    orientation=xmat_to_quat(self.data.site('end_effector').xmat),
                    top_image=obs['top'],
                    side_image=obs['side'],
                    handcam_left_image=obs['handcam_left'],
                    handcam_right_image=obs['handcam_right'],
                    grip=self.data.actuator('actuator8').ctrl,
                    joints=np.array([self.data.qpos[i] for i in range(7)])
                )
                self.outs.observation.write(observation, self.world.now_ts)


        self.renderer.close()
