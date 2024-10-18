import asyncio
import logging
import signal
import time
from typing import List, Tuple

import click
import hydra
from omegaconf import DictConfig


from control import ControlSystem, utils, World, MainThreadWorld
from geom import Quaternion, Transform3D
from hardware import Franka, DHGripper, sl_camera
from tools import rerun as rr_tools
from tools.dataset_dumper import DatasetDumper
from tools.inference import Inference

import curses

logging.basicConfig(level=logging.INFO,
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("policy_runner.log", mode="w")])

class PolicyRunnerSystem(ControlSystem):
    def __init__(self, world: World):
        super().__init__(world, inputs=[], outputs=["start_policy", "stop_policy"])

    def run(self):
        policy_running = False
        while not self.should_stop:
            user_input = input()
            if user_input == 's':
                if not policy_running:
                    print('Started policy execution')
                    policy_running = True
                    self.outs.start_policy.write(True, self.world.now_ts)
                else:
                    print('Stopped policy execution')
                    policy_running = False
                    self.outs.stop_policy.write(True, self.world.now_ts)


@hydra.main(version_base=None, config_path="configs", config_name="policy_runner")
def main(cfg: DictConfig):
    world = MainThreadWorld()
    franka = Franka(world,
                    cfg.franka.ip,
                    cfg.franka.relative_dynamics_factor,
                    cfg.franka.gripper_force,
                    reporting_frequency=cfg.franka.reporting_frequency)

    policy_runner = PolicyRunnerSystem(world)

    cam = sl_camera.SLCamera(world, view=sl_camera.sl.VIEW.SIDE_BY_SIDE,
                             fps=15, resolution=sl_camera.sl.RESOLUTION.VGA)

    inference = Inference(world, cfg.inference)

    # Connect inputs
    inference.ins.image = cam.outs.record
    inference.ins.robot_joints = franka.outs.joint_positions
    inference.ins.robot_position = franka.outs.position
    inference.ins.ext_force_ee = franka.outs.ext_force_ee
    inference.ins.ext_force_base = franka.outs.ext_force_base
    inference.ins.start = policy_runner.outs.start_policy
    inference.ins.stop = policy_runner.outs.stop_policy

    # Connect outputs
    franka.ins.target_position = inference.outs.target_robot_position

    gripper = DHGripper(world, "/dev/ttyUSB0")
    gripper.ins.grip = inference.outs.target_grip
    inference.ins.grip = gripper.outs.grip

    if cfg.rerun:
        connect = cfg.rerun if ':' in cfg.rerun else None
        save_path = None if ':' in cfg.rerun else cfg.rerun
        rr = rr_tools.Rerun(world, "policy_runner",
                            connect=connect,
                            save_path=save_path,
                            inputs={
                                "image": rr_tools.log_image,
                                "input/ext_force_ee": rr_tools.log_array,
                                "input/ext_force_base": rr_tools.log_array,
                                "input/robot_position": rr_tools.log_transform,
                                "input/robot_joints": rr_tools.log_array,
                                "input/grip": rr_tools.log_scalar,
                                "output/target_robot_position": rr_tools.log_transform,
                                "output/target_grip": rr_tools.log_scalar,
                            })
        @utils.map_port
        def image(record):
            return record.image
        rr.ins.image = image(cam.outs.record)
        rr.ins['input/ext_force_ee'] = franka.outs.ext_force_ee
        rr.ins['input/ext_force_base'] = franka.outs.ext_force_base
        rr.ins['input/robot_position'] = franka.outs.position
        rr.ins['input/robot_joints'] = franka.outs.joint_positions
        rr.ins['input/grip'] = gripper.outs.grip
        rr.ins['output/target_robot_position'] = inference.outs.target_robot_position
        rr.ins['output/target_grip'] = inference.outs.target_grip

    world.run()


if __name__ == "__main__":
    main()
