from typing import Optional

from control.utils import Throttler
import ironic as ir
from geom import Transform3D
from simulator.mujoco.sim import InverseKinematics, MujocoRenderer, MujocoSimulator


@ir.ironic_system(
    input_ports=["actuator_values", "gripper_target_grasp", "reset", "robot_target_position"],
    output_ports=["images"],
    output_props=[
        "robot_position",
        "grip",
        "joints",
        "ext_force_ee",
        "ext_force_base",
        "actuator_values",
    ])
class MujocoSimulatorCS(ir.ControlSystem):
    def __init__(
            self,
            simulator: MujocoSimulator,
            simulation_rate: float = 1 / 500,
            render_rate: float = 1 / 60,
            renderer: Optional[MujocoRenderer] = None,
            inverse_kinematics: Optional[InverseKinematics] = None,
    ):
        super().__init__()
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

    @ir.out_property
    async def robot_position(self):
        return ir.Message(
            Transform3D(
                translation=self.simulator.robot_position.translation,
                quaternion=self.simulator.robot_position.quaternion
            ),
            self.ts
        )

    @ir.out_property
    async def grip(self):
        return ir.Message(self.simulator.grip, self.ts)

    @ir.out_property
    async def joints(self):
        return ir.Message(self.simulator.joints, self.ts)

    @ir.out_property
    async def ext_force_ee(self):
        return ir.Message(self.simulator.ext_force_ee, self.ts)

    @ir.out_property
    async def ext_force_base(self):
        return ir.Message(self.simulator.ext_force_base, self.ts)

    @ir.out_property
    async def actuator_values(self):
        return ir.Message(self.simulator.actuator_values, self.ts)

    @property
    def ts(self):
        return self.simulator.ts

    def simulate(self):
        self.simulator.step()

    async def render(self):
        if self.renderer is not None:
            images = self.renderer.render_frames()
            await self.outs.images.write_message(images)

    def _init_position(self):
        self.simulator.reset()

    def _init_renderer(self):
        if self.renderer is not None:
            self.renderer.initialize()

    @ir.on_message('reset')
    async def on_reset(self, _message: ir.Message):
        self._init_position()

    @ir.on_message('gripper_target_grasp')
    async def on_gripper_target_grasp(self, message: ir.Message):
        self.simulator.set_grip(message.data)

    @ir.on_message('robot_target_position')
    async def on_robot_target_position(self, message: ir.Message):
        actuator_values = self.inverse_kinematics.recalculate_ik(message.data)

        if actuator_values is not None:
            self.simulator.set_actuator_values(actuator_values)

    @ir.on_message('actuator_values')
    async def on_actuator_values(self, message: ir.Message):
        self.simulator.set_actuator_values(message.data)

    async def setup(self):
        self._init_position()
        self._init_renderer()

    async def step(self):
        for _ in range(self.do_simulation()):
            self.simulate()

        if self.do_render():
            await self.render()

    async def cleanup(self):
        if self.renderer is not None:
            self.renderer.close()
