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

    def _control(self):
        track_but, untrack_but, grasp = False, False, False
        robot_t = None
        teleop_t = None
        offset = None
        is_tracking = False

        while True:
            input_name, value = yield
            if input_name == "teleop_transform":
                teleop_t = value
            elif input_name == "teleop_buttons":
                track_but, untrack_but, grasp_but, open_but = value
                grasp = (grasp and not open_but) or (not grasp and grasp_but)
            elif input_name == "robot_transform":
                robot_t = value

            if track_but:
                # Note that translation and rotation offsets are independent
                offset = Transform(-teleop_t.translation + robot_t.translation,
                                   q_mul(q_inv(teleop_t.quaternion), robot_t.quaternion))
                is_tracking = True
            elif untrack_but:
                is_tracking = False
                offset = None
            elif is_tracking:
                self.outs.gripper_grasped.write(grasp)
                target = Transform(teleop_t.translation + offset.translation,
                                   q_mul(teleop_t.quaternion, offset.quaternion))
                self.outs.transform.write(target)


def main():
    # webxr = WebXR(port=5005)
    # franka = Franka("172.168.0.2", 0.2, 0.4, franky.RealtimeConfig.Ignore)
    # teleop = TeleopSystem()

    # teleop.ins.teleop_transform = webxr.outs.transform
    # teleop.ins.teleop_buttons = webxr.outs.buttons
    # teleop.ins.robot_transform = franka.outs.transform

    # franka.ins.transform = teleop.outs.transform
    # franka.ins.gripper_grasped = teleop.outs.gripper_grasped

    webxr = WebXR(port=5005)
    logger = utils.Map(inputs=["transform", "buttons"], default=lambda n, v : print(f'{n}: {v}'))
    logger.ins.transform = webxr.outs.transform
    logger.ins.buttons = webxr.outs.buttons
    logger.start()
    webxr.start()  # WebXR must start last, as it is a blocking call


if __name__ == "__main__":
    main()
