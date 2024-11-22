import asyncio
import logging
from typing import List, Tuple

import numpy as np
import hydra
from omegaconf import DictConfig
import yappi

import hardware
import ironic as ir
from geom import Quaternion, Transform3D
from hardware import Franka, DHGripper
from tools.dataset_dumper import DatasetDumper
from webxr import WebXR

logging.basicConfig(level=logging.INFO,
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("teleop.log", mode="w")])


@ir.ironic_system(input_ports=["teleop_transform", "teleop_buttons"],
                  input_props=["robot_position"],
                  output_ports=["robot_target_position", "gripper_target_grasp", "start_tracking", "stop_tracking"])
class TeleopSystem(ir.ControlSystem):
    def __init__(self):
        super().__init__()
        self.teleop_t = None
        self.offset = None
        self.is_tracking = False
        self.fps = ir.utils.FPSCounter("Teleop")

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

    @ir.on_message("teleop_transform")
    async def handle_teleop_transform(self, message: ir.Message):
        self.teleop_t = self._parse_position(message.data)
        self.fps.tick()

        if self.is_tracking and self.offset is not None:
            target = Transform3D(
                self.teleop_t.translation + self.offset.translation,
                self.teleop_t.quaternion * self.offset.quaternion
            )
            await self.outs.robot_target_position.write(ir.Message(target, message.timestamp))

    @ir.on_message("teleop_buttons")
    async def handle_teleop_buttons(self, message: ir.Message):
        track_but, untrack_but, grasp_but = self._parse_buttons(message.data)

        if self.is_tracking:
            await self.outs.gripper_target_grasp.write(ir.Message(grasp_but, message.timestamp))

        if track_but:
            # Note that translation and rotation offsets are independent
            if self.teleop_t is not None:
                robot_t = (await self.ins.robot_position()).data
                self.offset = Transform3D(
                    -self.teleop_t.translation + robot_t.translation,
                    self.teleop_t.quaternion.inv * robot_t.quaternion
                )
            if not self.is_tracking:
                logging.info('Started tracking')
                self.is_tracking = True
                await self.outs.start_tracking.write(ir.Message(None, message.timestamp))
        elif untrack_but:
            if self.is_tracking:
                logging.info('Stopped tracking')
                self.is_tracking = False
                self.offset = None
                await self.outs.stop_tracking.write(ir.Message(None, message.timestamp))


@hydra.main(version_base=None, config_path="configs", config_name="teleop")
def main(cfg: DictConfig):
    asyncio.run(main_async(cfg))


async def main_async(cfg: DictConfig):
    webxr = WebXR(port=cfg.webxr.port)
    franka = Franka(cfg.franka.ip, cfg.franka.relative_dynamics_factor, cfg.franka.gripper_force)
    teleop = TeleopSystem()

    components = [webxr, franka, teleop]

    # Connect teleop system
    teleop.bind(
        teleop_transform=webxr.outs.transform,
        teleop_buttons=webxr.outs.buttons,
        robot_position=franka.outs.position
    )
    franka.bind(target_position=teleop.outs.robot_target_position)

    gripper = None
    if 'dh_gripper' in cfg:
        gripper = DHGripper(cfg.dh_gripper).bind(grip=teleop.outs.gripper_target_grasp)
        components.append(gripper)

    cam = hardware.from_config.sl_camera(cfg.camera)
    components.append(cam)

    if cfg.data_output_dir is not None:
        properties_to_dump = ir.utils.properties_dict(
            robot_joints=franka.outs.joint_positions,
            robot_position_translation=ir.utils.map_property(lambda t: t.translation, franka.outs.position),
            robot_position_quaternion=ir.utils.map_property(lambda t: t.quaternion, franka.outs.position),
            ext_force_ee=franka.outs.ext_force_ee,
            ext_force_base=franka.outs.ext_force_base,
            grip=gripper.outs.grip if gripper else None
        )

        components.append(DatasetDumper(cfg.data_output_dir).bind(
            image=cam.outs.frame,
            start_episode=teleop.outs.start_tracking,
            end_episode=teleop.outs.stop_tracking,
            target_grip=teleop.outs.gripper_target_grasp,
            target_robot_position=teleop.outs.robot_target_position,
            robot_data=properties_to_dump
        ))

    if cfg.profile:
        yappi.set_clock_type("cpu")
        yappi.start(profile_threads=False)

    def profile_cleanup():
        if cfg.profile:
            yappi.stop()
            yappi.get_func_stats().save("func.pstat", type='pstat')
            yappi.get_func_stats().save("func.ystat")
            with open("thread.ystat", "w") as f:
                yappi.get_thread_stats().print_all(out=f)

    system = ir.compose(*components)
    await ir.utils.run_gracefully(system, extra_cleanup_fn=profile_cleanup)


if __name__ == "__main__":
    main()
