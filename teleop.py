import asyncio
import logging
from typing import List, Tuple

import numpy as np

from control import ControlSystem, utils
from geom import Quaternion, Transform3D
from hardware import Franka, Kinova
from hardware import DHGripper
from webxr import WebXR

logging.basicConfig(level=logging.INFO,
                     handlers=[logging.StreamHandler(),
                               logging.FileHandler("teleop.log", mode="w")])

class TeleopSystem(ControlSystem):
    def __init__(self):
        super().__init__(
            inputs=["teleop_transform", "teleop_buttons", "robot_position"],
            outputs=["robot_target_position", "gripper_target_grasp"])

    @classmethod
    def _parse_position(cls, value: Transform3D) -> Transform3D:
        pos = np.array([value.translation[2], value.translation[0], value.translation[1]])
        quat = Quaternion(value.quaternion[0], value.quaternion[3], value.quaternion[1], value.quaternion[2])

        # Don't ask my why these transformations, I just got them
        # Rotate quat 90 degrees around Y axis
        res_quat = quat
        rotation_y_90 = Quaternion(np.cos(-np.pi/4), 0, np.sin(-np.pi/4), 0)
        res_quat = rotation_y_90 * quat
        res_quat = Quaternion(-res_quat[0], res_quat[1], res_quat[2], res_quat[3])
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
            if input_name == "robot_position":
                robot_t = value
            elif input_name == "teleop_transform":
                teleop_t = self._parse_position(value)
                if is_tracking:
                    if offset is not None:
                        target = Transform3D(teleop_t.translation + offset.translation,
                                             teleop_t.quaternion * offset.quaternion)
                        await self.outs.robot_target_position.write(target)
            elif input_name == "teleop_buttons":
                track_but, untrack_but, grasp_but = self._parse_buttons(value)

                if is_tracking: await self.outs.gripper_target_grasp.write(grasp_but)

                if track_but:
                    # Note that translation and rotation offsets are independent
                    if teleop_t is not None and robot_t is not None:
                        offset = Transform3D(-teleop_t.translation + robot_t.translation,
                                             teleop_t.quaternion.inv * robot_t.quaternion)
                    if not is_tracking:
                        logging.info('Started tracking')
                        is_tracking = True
                elif untrack_but:
                    if is_tracking:
                        logging.info('Stopped tracking')
                        is_tracking = False
                        offset = None

async def main():
    webxr = WebXR(port=5005)
    franka = Franka("172.168.0.2", 0.2, 0.4)
    # kinova = Kinova('192.168.1.10')
    gripper = DHGripper("/dev/ttyUSB0")
    teleop = TeleopSystem()

    teleop.ins.teleop_transform = webxr.outs.transform
    teleop.ins.teleop_buttons = webxr.outs.buttons
    teleop.ins.robot_position = franka.outs.position

    gripper.ins.grip = teleop.outs.gripper_target_grasp
    franka.ins.target_position = teleop.outs.robot_target_position

    await asyncio.gather(teleop.run(), webxr.run(), franka.run(), gripper.run())


if __name__ == "__main__":
    asyncio.run(main())
