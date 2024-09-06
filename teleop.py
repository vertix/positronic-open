import asyncio
import logging
from typing import List, Tuple

import click
import numpy as np
import yappi

from control import ControlSystem, utils, World, MainThreadWorld
from geom import Quaternion, Transform3D
from hardware import Franka, DHGripper, sl_camera
from tools import rerun as rr_tools
from tools.lerobot import LerobotDatasetDumper
from webxr import WebXR

logging.basicConfig(level=logging.INFO,
                     handlers=[logging.StreamHandler(),
                               logging.FileHandler("teleop.log", mode="w")])

class TeleopSystem(ControlSystem):
    def __init__(self, world: World):
        super().__init__(
            world,
            inputs=["teleop_transform", "teleop_buttons", "robot_position"],
            outputs=["robot_target_position", "gripper_target_grasp", "start_tracking", "stop_tracking"])

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

    def run(self):
        track_but, untrack_but, grasp_but = 0., 0., 0.
        robot_t = None
        teleop_t = None
        offset = None
        is_tracking = False

        fps = utils.FPSCounter("Teleop")
        for input_name, _ts, value in self.ins.read():
            if input_name == "robot_position":
                robot_t = value
            elif input_name == "teleop_transform":
                teleop_t = self._parse_position(value)
                fps.tick()
                if is_tracking and offset is not None:
                    target = Transform3D(teleop_t.translation + offset.translation,
                                         teleop_t.quaternion * offset.quaternion)
                    self.outs.robot_target_position.write(target)
            elif input_name == "teleop_buttons":
                track_but, untrack_but, grasp_but = self._parse_buttons(value)

                if is_tracking: self.outs.gripper_target_grasp.write(grasp_but)

                if track_but:
                    # Note that translation and rotation offsets are independent
                    if teleop_t is not None and robot_t is not None:
                        offset = Transform3D(-teleop_t.translation + robot_t.translation,
                                             teleop_t.quaternion.inv * robot_t.quaternion)
                    if not is_tracking:
                        logging.info('Started tracking')
                        is_tracking = True
                        self.outs.start_tracking.write(True)
                elif untrack_but:
                    if is_tracking:
                        logging.info('Stopped tracking')
                        is_tracking = False
                        offset = None
                        self.outs.stop_tracking.write(True)


def main(rerun, dh_gripper):
    world = MainThreadWorld()
    webxr = WebXR(world, port=5005)
    franka = Franka(world, "172.168.0.2", 0.4, 0.4, reporting_frequency=None)

    teleop = TeleopSystem(world)

    teleop.ins.teleop_transform = webxr.outs.transform
    teleop.ins.teleop_buttons = webxr.outs.buttons
    teleop.ins.robot_position = franka.outs.position
    franka.ins.target_position = teleop.outs.robot_target_position

    if dh_gripper:
        gripper = DHGripper("/dev/ttyUSB0")
        gripper.ins.grip = teleop.outs.gripper_target_grasp

    cam = sl_camera.SLCamera(world, fps=15, resolution=sl_camera.sl.RESOLUTION.VGA)

    # TODO: Move it under command line flags
    data_dumper = LerobotDatasetDumper(world, '_dataset')
    data_dumper.ins.image = cam.outs.record
    data_dumper.ins.robot_joints = franka.outs.joint_positions
    data_dumper.ins.robot_position = franka.outs.position
    data_dumper.ins.ext_force_ee = franka.outs.ext_force_ee
    data_dumper.ins.ext_force_base = franka.outs.ext_force_base
    data_dumper.ins.start_episode = teleop.outs.start_tracking
    data_dumper.ins.end_episode = teleop.outs.stop_tracking

    if rerun:
        rr = rr_tools.Rerun(world, "teleop",
                         connect="127.0.0.1:9876",
                         inputs={"ext_force_ee": rr_tools.log_array, 'ext_force_base': rr_tools.log_array, 'image': rr_tools.log_image})
        @utils.map_port
        def image(record):
            return record.image.get_data()[:, :, :3]
        rr.ins.image = image(cam.outs.record)

        rr.ins.ext_force_ee = franka.outs.ext_force_ee
        rr.ins.ext_force_base = franka.outs.ext_force_base

    yappi.set_clock_type("cpu")
    yappi.start(profile_threads=False)
    try:
        world.run()
    finally:
        print("Program interrupted by user, exiting...")
        yappi.stop()
        yappi.get_func_stats().save("func.pstat", type='pstat')
        yappi.get_func_stats().save("func.ystat")
        with open("thread.ystat", "w") as f:
            yappi.get_thread_stats().print_all(out=f)
        print("Program exited, stats saved")


@click.command()
@click.option("--rerun", is_flag=True, default=False, help="Start logging into Rerun")
@click.option("--dh_gripper", is_flag=True, default=False, help="Use DH gripper")
def cli(rerun, dh_gripper):
    asyncio.run(main(rerun, dh_gripper))


if __name__ == "__main__":
    cli()
