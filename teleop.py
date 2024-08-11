import asyncio

import franky

from control import ControlSystem, utils
from geom import Transform, q_mul, q_inv
from hardware import Franka
from webxr import WebXR


class TeleopSystem(ControlSystem):
    def __init__(self):
        super().__init__(
            inputs=["teleop_transform", "teleop_buttons", "robot_transform"],
            outputs=["transform", "gripper_grasped"])

    async def run(self):
        track_but, untrack_but, grasp = False, False, False
        robot_t = None
        teleop_t = None
        offset = None
        is_tracking = False

        async for input_name, value in self.ins.read():
            if input_name == "robot_transform":
                robot_t = value
            else:
                if input_name == "teleop_transform":
                    teleop_t = value
                elif input_name == "teleop_buttons":
                    track_but, untrack_but, grasp_but, open_but = value
                    grasp = (grasp and not open_but) or (not grasp and grasp_but)

                if track_but:
                    # Note that translation and rotation offsets are independent
                    offset = Transform(-teleop_t.translation + robot_t.translation,
                                    q_mul(q_inv(teleop_t.quaternion), robot_t.quaternion))
                    is_tracking = True
                elif untrack_but:
                    is_tracking = False
                    offset = None
                elif is_tracking:
                    # await self.outs.gripper_grasped.write(grasp)
                    target = Transform(teleop_t.translation + offset.translation,
                                    q_mul(teleop_t.quaternion, offset.quaternion))
                    await self.outs.transform.write(target)



async def main():
    webxr = WebXR(port=5005)
    franka = Franka("172.168.0.2", 0.2, 0.4, franky.RealtimeConfig.Ignore)
    teleop = TeleopSystem()

    teleop.ins.teleop_transform = webxr.outs.transform
    teleop.ins.teleop_buttons = webxr.outs.buttons
    teleop.ins.robot_transform = franka.outs.transform

    franka.ins.transform = teleop.outs.transform
    franka.ins.gripper_grasped = teleop.outs.gripper_grasped

    await asyncio.gather(teleop.run(), webxr.run(), franka.run())


if __name__ == "__main__":
    asyncio.run(main())
