from typing import Optional

from control import ControlSystem, control_system
from control.system import output_property_custom_time
from control.utils import Throttler
from control.world import World
from geom import Transform3D
from simulator.mujoco.sim import InverseKinematics, MujocoRenderer, MujocoSimulator


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
        self.do_simulation = Throttler(every_sec=simulation_rate)
        self.simulation_rate = simulation_rate
        self.pending_actions = []

        self.renderer = renderer

        if renderer is not None:
            self.do_render = Throttler(every_sec=render_rate)
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
