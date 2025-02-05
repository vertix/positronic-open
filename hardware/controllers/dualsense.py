from pydualsense import pydualsense, DSState

import ironic as ir
import geom
from tools.buttons import ButtonHandler


@ir.ironic_system(input_props=['robot_position'],
                 output_ports=[
                      "robot_target_position",
                      "gripper_target_grasp",
                      "start_tracking",
                      "stop_tracking",
                      "start_recording",
                      "stop_recording",
                      "reset"
                  ])
class DualSenseCS(ir.ControlSystem):
    last_state: DSState

    def __init__(self):
        super().__init__()
        self.controller = pydualsense()
        self.last_state = None
        self.last_timestamp = None
        self.tracking = False
        self.recording = False
        self.fps = ir.utils.FPSCounter('DualSense')
        self.button_handler = ButtonHandler()

    async def setup(self):
        self.controller.init()

    async def step(self):
        self.last_state = self.controller.state
        self.last_timestamp = ir.system_clock()
        self.fps.tick()

        self.button_handler.update_buttons(
            {
                'tracking': self.last_state.square,
                'recording': self.last_state.circle,
                'trigger': self.last_state.L2,
                'pitch': self.last_state.gyro.Pitch,
                'yaw': self.last_state.gyro.Yaw,
                'roll': self.last_state.gyro.Roll
            }
        )

        if self.button_handler.is_pressed('tracking'):
            await self.switch_tracking()
        if self.button_handler.is_pressed('recording'):
            await self.switch_recording()

        target_position = geom.Transform3D(
            translation=[0, 0, 0],
            quaternion=geom.Quaternion.from_euler([self.button_handler.get_value('roll'), self.button_handler.get_value('pitch'), self.button_handler.get_value('yaw')])
        )
        await self.outs.robot_target_position.write(ir.Message(data=target_position, timestamp=ir.system_clock()))

        target_grip = self.last_state.L2
        await self.outs.gripper_target_grasp.write(ir.Message(data=target_grip, timestamp=ir.system_clock()))

        return ir.State.ALIVE


    async def cleanup(self):
        self.controller.close()

    async def switch_tracking(self):
        self.tracking = not self.tracking
        if self.tracking:
            await self.outs.start_tracking.write(ir.Message(data=self.tracking, timestamp=ir.system_clock()))
        else:
            await self.outs.stop_tracking.write(ir.Message(data=self.tracking, timestamp=ir.system_clock()))

    async def switch_recording(self):
        self.recording = not self.recording
        if self.recording:
            await self.outs.start_recording.write(ir.Message(data=self.recording, timestamp=ir.system_clock()))
        else:
            await self.outs.stop_recording.write(ir.Message(data=self.recording, timestamp=ir.system_clock()))
