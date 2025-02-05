import numpy as np

import ironic as ir


@ir.ironic_system(
    output_props=['robot_position', 'robot_state', 'joint_positions', 'ext_force_base', 'ext_force_ee', 'grip'],
    input_ports=['target_position', 'target_grip']
)
class UmiCS(ir.ControlSystem):
    def __init__(self):
        super().__init__()
        self.target_position = None
        self.target_grip = None

    @ir.out_property
    async def robot_position(self):
        # TODO: here we will put registered robot position
        return ir.Message(data=self.target_position, timestamp=ir.system_clock())

    @ir.out_property
    async def robot_state(self):
        # TODO: here we could put robot state like force sensors, IMU, etc.
        return ir.Message(data=np.array([0, 0, 0]), timestamp=ir.system_clock())

    @ir.out_property
    async def ext_force_base(self):
        # TODO: here we could put robot state like force sensors, IMU, etc.
        return ir.Message(data=np.array([0, 0, 0]), timestamp=ir.system_clock())

    @ir.out_property
    async def ext_force_ee(self):
        # TODO: here we could put robot state like force sensors, IMU, etc.
        return ir.Message(data=np.array([0, 0, 0]), timestamp=ir.system_clock())

    @ir.out_property
    async def grip(self):
        return ir.Message(data=0, timestamp=ir.system_clock())

    @ir.out_property
    async def joint_positions(self):
        return ir.Message(data=np.array([0, 0, 0]), timestamp=ir.system_clock())

    @ir.on_message('target_position')
    async def on_target_position(self, message: ir.Message):
        self.target_position = message.data

    @ir.on_message('target_grip')
    async def on_target_grip(self, message: ir.Message):
        self.target_grip = message.data
