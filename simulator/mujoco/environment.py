from typing import Optional

import mujoco
import numpy as np
from dm_control import mujoco as dm_mujoco
from dm_control.utils import inverse_kinematics as ik

from control import ControlSystem, control_system
from control.system import output_property_custom_time
from control.utils import IntervalChecker
from control.world import World
from geom import Transform3D
from simulator.mujoco.sim import MujocoRenderer, MujocoSimulator


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


@control_system(
    inputs=["actuator_values", "target_grip", "reset", "target_robot_position"],
    outputs=["images"],  
    output_props=[
        "robot_position",
        "grip",
        "joints",
        "ext_force_ee",
        "ext_force_base",
        "actuator_values",
    ])
class MujocoSimulatorCS(ControlSystem):
    def __init__(
            self,
            world: World,
            simulator: MujocoSimulator,
            simulation_rate: float = 1 / 500,
            render_rate: float = 1 / 60,
            renderer: Optional[MujocoRenderer] = None,
            inverse_kinematics: Optional[InverseKinematics] = None,

    ):
        super().__init__(world)
        self.simulator = simulator
        self.do_simulation = IntervalChecker(interval=simulation_rate * 1000, time_fn=lambda: self.world.now_ts)
        self.simulation_rate = simulation_rate
        self.pending_actions = []

        self.renderer = renderer

        if renderer is not None:
            self.do_render = IntervalChecker(interval=render_rate * 1000, time_fn=lambda: self.world.now_ts)
        else:
            self.do_render = lambda: False

        self.inverse_kinematics = inverse_kinematics

    @output_property_custom_time('robot_position')
    def robot_position(self):
        return Transform3D(
            translation=self.simulator.robot_position.translation,
            quaternion=self.simulator.robot_position.quaternion
        ), self.ts

    @output_property_custom_time('grip')
    def grip(self):
        return self.simulator.grip, self.ts

    @output_property_custom_time('joints')
    def joints(self):
        return self.simulator.joints, self.ts

    @output_property_custom_time('ext_force_ee')
    def ext_force_ee(self):
        return self.simulator.ext_force_ee, self.ts

    @output_property_custom_time('ext_force_base')
    def ext_force_base(self):
        return self.simulator.ext_force_base, self.ts

    @output_property_custom_time('actuator_values')
    def actuator_values(self):
        return self.simulator.actuator_values, self.ts

    @property
    def ts(self):
        return self.simulator.ts

    def simulate(self):
        self.simulator.step()

    def render(self):
        if self.renderer is not None:
            images = self.renderer.render_frames()
            self.outs.images.write(images, self.ts)

    def _init_position(self):
        self.simulator.reset()
    
    def _init_renderer(self):
        if self.renderer is not None:
            self.renderer.initialize()

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

        target_robot_position = self.ins.target_robot_position.read_nowait()
        if target_robot_position is not None:
            ts, target_robot_position_value = target_robot_position
            actuator_values = self.inverse_kinematics.recalculate_ik(target_robot_position_value)
            if actuator_values is not None:
                self.simulator.set_actuator_values(actuator_values)

    def run(self):
        self._init_position()
        self._init_renderer()

        while not self.world.should_stop:           
            self._handle_inputs()

            for _ in range(self.do_simulation()):
                self.simulate()
            
            if self.do_render():
                self.render()
                

        if self.renderer is not None:
            self.renderer.close()
