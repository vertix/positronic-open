import asyncio
from typing import List, Tuple

import franky
import numpy as np

from control import ControlSystem, utils
from geom import Transform3D, q_mul, q_inv
from hardware import Franka
from hardware import DHGripper
from webxr import WebXR


class TeleopSystem(ControlSystem):
    def __init__(self):
        super().__init__(
            inputs=["teleop_transform", "teleop_buttons", "robot_transform"],
            outputs=["transform", "gripper_width"])

    @classmethod
    def _parse_position(cls, value: Transform3D) -> Transform3D:
        pos = np.array([value.translation[2], value.translation[0], value.translation[1]])
        quat = np.array([value.quaternion[0], value.quaternion[3], value.quaternion[1], value.quaternion[2]])

        # Don't ask my why these transformations, I just got them
        # Rotate quat 90 degrees around Y axis
        rotation_y_90 = np.array([np.cos(-np.pi/4), 0, np.sin(-np.pi/4), 0])
        res_quat = q_mul(rotation_y_90, quat)
        res_quat = np.array([-res_quat[0], res_quat[1], res_quat[2], res_quat[3]])
        return Transform3D(pos, res_quat)

    @classmethod
    def _parse_buttons(cls, value: List[float]) -> Tuple[float, float, float]:
        if len(value) > 6:
            but = value[4], value[5], value[0]
        else:
            but = 0., 0., 0.
        return but

    async def run(self):
        track_but, untrack_but, grasp_but = 0., 0., 0.
        robot_t = None
        teleop_t = None
        offset = None
        is_tracking = False

        async for input_name, value in self.ins.read():
            if input_name == "robot_transform":
                robot_t = value
            else:
                if input_name == "teleop_transform":
                    teleop_t = self._parse_position(value)
                elif input_name == "teleop_buttons":
                    track_but, untrack_but, grasp_but = self._parse_buttons(value)

                if track_but:
                    # Note that translation and rotation offsets are independent
                    if teleop_t is not None and robot_t is not None:
                        offset = Transform3D(-teleop_t.translation + robot_t.translation,
                                             q_mul(q_inv(teleop_t.quaternion), robot_t.quaternion))
                    is_tracking = True
                elif untrack_but:
                    is_tracking = False
                    offset = None
                elif is_tracking:
                    await self.outs.gripper_width.write(grasp_but)
                    if offset is not None:
                        target = Transform3D(teleop_t.translation + offset.translation,
                                             q_mul(teleop_t.quaternion, offset.quaternion))
                        await self.outs.transform.write(target)


async def main():
    webxr = WebXR(port=5005)
    # franka = Franka("172.168.0.2", 0.2, 0.4, franky.RealtimeConfig.Ignore)
    gripper = DHGripper("/dev/ttyUSB0")
    teleop = TeleopSystem()

    # teleop.ins.teleop_transform = webxr.outs.transform
    teleop.ins.teleop_buttons = webxr.outs.buttons
    # teleop.ins.robot_transform = franka.outs.transform

    gripper.ins.grip = teleop.outs.gripper_width

    # franka.ins.transform = teleop.outs.transform
    # franka.ins.gripper_grasped = teleop.outs.gripper_width

    await asyncio.gather(teleop.run(), webxr.run(), gripper.run())


if __name__ == "__main__":
    asyncio.run(main())
