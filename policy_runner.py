import logging

import hydra
from omegaconf import DictConfig
import rerun as rr

from control import MainThreadWorld
from control.utils import control_system_fn
from hardware import Franka, DHGripper, sl_camera
from tools.inference import Inference

logging.basicConfig(level=logging.INFO,
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("policy_runner.log", mode="w")])

@control_system_fn(outputs=["start_policy", "stop_policy"])
def PolicyRunnerSystem(ins, outs):
    policy_running = False
    while not ins.should_stop:
        user_input = input()
        if user_input == 's':
            if not policy_running:
                print('Started policy execution')
                policy_running = True
                outs.start_policy.write(True, ins.now_ts)
            else:
                print('Stopped policy execution')
                policy_running = False
                outs.stop_policy.write(True, ins.now_ts)


@hydra.main(version_base=None, config_path="configs", config_name="policy_runner")
def main(cfg: DictConfig):
    if cfg.rerun:
        rr.init("inference", spawn=False)
        if ':' in cfg.rerun:
            rr.connect(cfg.rerun)
        elif cfg.rerun is not None:
            rr.save(cfg.rerun)

    world = MainThreadWorld()
    franka = Franka(world,
                    cfg.franka.ip,
                    cfg.franka.relative_dynamics_factor,
                    cfg.franka.gripper_force)

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

    world.run()
    if cfg.rerun:
        rr.disconnect()
    print('Finished')


if __name__ == "__main__":
    main()
