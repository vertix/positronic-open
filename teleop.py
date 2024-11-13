import asyncio
import logging
from typing import List, Tuple

import numpy as np
import hydra
from omegaconf import DictConfig
import yappi

from control import ControlSystem, utils, MainThreadWorld, control_system
from geom import Quaternion, Transform3D
from hardware import Franka, DHGripper, sl_camera
from tools.dataset_dumper import DatasetDumper
from webxr import WebXR

logging.basicConfig(level=logging.INFO,
                     handlers=[logging.StreamHandler(),
                               logging.FileHandler("teleop.log", mode="w")])


@control_system(inputs=["teleop_transform", "teleop_buttons"],
                input_props=["robot_position"],
                outputs=["robot_target_position", "gripper_target_grasp", "start_tracking", "stop_tracking"])
class TeleopSystem(ControlSystem):
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
        teleop_t = None
        offset = None
        is_tracking = False

        fps = utils.FPSCounter("Teleop")
        def on_teleop_transform(value, _ts):
            teleop_t = self._parse_position(value)
            fps.tick()
            if is_tracking and offset is not None:
                target = Transform3D(teleop_t.translation + offset.translation,
                                     teleop_t.quaternion * offset.quaternion)
                self.outs.robot_target_position.write(target, self.world.now_ts)

        def on_teleop_buttons(value, _ts):
            track_but, untrack_but, grasp_but = self._parse_buttons(value)

            if is_tracking: self.outs.gripper_target_grasp.write(grasp_but, self.world.now_ts)

            if track_but:
                # Note that translation and rotation offsets are independent
                if teleop_t is not None:
                    robot_t, _ts = self.ins.robot_position()
                    offset = Transform3D(-teleop_t.translation + robot_t.translation,
                                            teleop_t.quaternion.inv * robot_t.quaternion)
                if not is_tracking:
                    logging.info('Started tracking')
                    is_tracking = True
                    self.outs.start_tracking.write(True, self.world.now_ts)
            elif untrack_but:
                if is_tracking:
                    logging.info('Stopped tracking')
                    is_tracking = False
                    offset = None
                    self.outs.stop_tracking.write(True, self.world.now_ts)

        with self.ins.subscribe(teleop_transform=on_teleop_transform,
                                teleop_buttons=on_teleop_buttons):
            for _ in self.ins.read(): pass


@hydra.main(version_base=None, config_path="configs", config_name="teleop")
def main(cfg: DictConfig):
    world = MainThreadWorld()
    webxr = WebXR(world, port=cfg.webxr.port)
    franka = Franka(world, cfg.franka.ip, cfg.franka.relative_dynamics_factor, cfg.franka.gripper_force)

    teleop = TeleopSystem(world)

    teleop.ins.teleop_transform = webxr.outs.transform
    teleop.ins.teleop_buttons = webxr.outs.buttons
    teleop.ins.robot_position = franka.outs.position
    franka.ins.target_position = teleop.outs.robot_target_position

    if 'dh_gripper' in cfg:
        gripper = DHGripper(world, cfg.dh_gripper)
        gripper.ins.grip = teleop.outs.gripper_target_grasp

    cam = sl_camera.SLCamera(world, view=sl_camera.sl.VIEW.SIDE_BY_SIDE,
                             fps=cfg.camera.fps, resolution=sl_camera.sl.RESOLUTION.VGA)

    if cfg.data_output_dir is not None:
        data_dumper = DatasetDumper(world, cfg.data_output_dir)
        data_dumper.ins.image = cam.outs.record
        data_dumper.ins.robot_joints = franka.outs.joint_positions
        data_dumper.ins.robot_position = franka.outs.position
        data_dumper.ins.ext_force_ee = franka.outs.ext_force_ee
        data_dumper.ins.ext_force_base = franka.outs.ext_force_base
        data_dumper.ins.start_episode = teleop.outs.start_tracking
        data_dumper.ins.end_episode = teleop.outs.stop_tracking
        data_dumper.ins.target_grip = teleop.outs.gripper_target_grasp
        data_dumper.ins.target_robot_position = teleop.outs.robot_target_position
        if 'dh_gripper' in cfg:
            data_dumper.ins.grip = gripper.outs.grip

    if cfg.profile:
        yappi.set_clock_type("cpu")
        yappi.start(profile_threads=False)

    try:
        world.run()
    finally:
        print("Program interrupted by user, exiting...")
        if cfg.profile:
            yappi.stop()
            yappi.get_func_stats().save("func.pstat", type='pstat')
            yappi.get_func_stats().save("func.ystat")
            with open("thread.ystat", "w") as f:
                yappi.get_thread_stats().print_all(out=f)
            print("Program exited, stats saved")
        else:
            print("Program exited")


if __name__ == "__main__":
    main()
