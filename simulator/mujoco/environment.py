from dataclasses import dataclass
from typing import Tuple, Optional
from threading import Lock

import mujoco
import numpy as np
from dm_control import mujoco as dm_mujoco
from dm_control.utils import inverse_kinematics as ik

from control import ControlSystem, control_system
from control.system import output_property_custom_time
from control.world import World
from geom import Transform3D
from simulator.mujoco.sim import MujocoRenderer, MujocoSimulator

mjc_lock = Lock()

@dataclass
class DesiredAction:
    position: np.ndarray
    orientation: np.ndarray
    grip: float


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


@control_system(
    inputs=["actuator_values", "target_grip", "reset"],
    outputs=["step_complete"],  
    output_props=[
        "robot_position",
        "grip",
        "joints",
        "ext_force_ee",
        "ext_force_base",
        "actuator_values",
        "ts"
    ])
class MujocoSimulatorCS(ControlSystem):
    def __init__(
            self,
            world: World,
            simulator: MujocoSimulator,
            simulation_rate: float = 1 / 500,
    ):
        super().__init__(world)
        self.simulator = simulator
        self.last_simulation_time = None
        self.simulation_rate = simulation_rate
        self.pending_actions = []

    @output_property_custom_time('robot_position')
    def robot_position(self):
        return Transform3D(
            translation=self.simulator.robot_position.translation,
            quaternion=self.simulator.robot_position.quaternion
        ), self.ts()    

    @output_property_custom_time('grip')
    def grip(self):
        return self.simulator.grip, self.ts()

    @output_property_custom_time('joints')
    def joints(self):
        return self.simulator.joints, self.ts()

    @output_property_custom_time('ext_force_ee')
    def ext_force_ee(self):
        return self.simulator.ext_force_ee, self.ts()

    @output_property_custom_time('ext_force_base')
    def ext_force_base(self):
        return self.simulator.ext_force_base, self.ts()

    @output_property_custom_time('actuator_values')
    def actuator_values(self):
        return self.simulator.actuator_values, self.ts()

    @output_property_custom_time('ts')
    def ts(self):
        return self.simulator.ts

    def simulate(self):
        with mjc_lock:
            self.last_simulation_time = self.world.now_ts
            self.simulator.step()
        self.outs.step_complete.write(True, self.ts())

    def _init_position(self):
        with mjc_lock:
            self.simulator.reset()
        self.last_simulation_time = self.world.now_ts

    def _handle_inputs(self):
        result = self.ins.actuator_values.read_nowait()
        if result is not None:
            ts, values = result
            if values is not None:
                self.simulator.set_actuator_values(values)
        grip = self.ins.target_grip.read_nowait()
        if grip is not None:
            ts, grip_value = grip
            self.simulator.set_grip(grip_value)
        reset = self.ins.reset.read_nowait()
        if reset is not None:
            self._init_position()

    def run(self):
        self._init_position()

        while not self.world.should_stop:
            time_since_last_sim = self.world.now_ts - self.last_simulation_time
            
            self._handle_inputs()

            if time_since_last_sim >= self.simulation_rate * 1000:
                self.simulate()


@control_system(inputs=["step_complete"], outputs=["images"])
class MujocoRendererCS(ControlSystem):
    def __init__(
            self,
            world: World,
            renderer: MujocoRenderer,
            render_resolution: Tuple[int, int] = (320, 240),
            max_fps: int = 60
    ):
        super().__init__(world)
        self.renderer = renderer
        self.render_resolution = render_resolution
        self.render_rate = 1 / max_fps
        self.last_render_time = -float('inf')


    def run(self):
        self.renderer.initialize()
        for ts, _ in self.ins.step_complete.read_until_stop():
            if ts - self.last_render_time < 0:
                # simulator reset
                self.last_render_time = -float('inf')

            if ts - self.last_render_time >= self.render_rate:
                self.last_render_time = ts

                with mjc_lock:
                    images = self.renderer.render_frames()
                self.outs.images.write(images, ts)

        self.renderer.close()

