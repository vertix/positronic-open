import asyncio
import logging
from typing import List

import hydra
from omegaconf import DictConfig

import hardware
import ironic as ir
from hardware import Franka, DHGripper
from simulator.mujoco.environment import MujocoSimulatorCS
from simulator.mujoco.sim import MujocoSimulator, MujocoRenderer, InverseKinematics
from tools.dataset_dumper import DatasetDumper
from webxr import WebXR
from simulator.mujoco.mujoco_gui import DearpyguiUi
import mujoco

logging.basicConfig(level=logging.INFO,
                   handlers=[logging.StreamHandler(),
                             logging.FileHandler("data_collection.log", mode="w")])

async def setup_mujoco(cfg: DictConfig):
    """Setup and return MuJoCo simulator components"""
    width = cfg.mujoco.camera_width
    height = cfg.mujoco.camera_height

    model = mujoco.MjModel.from_xml_path(cfg.mujoco.model_path)
    data = mujoco.MjData(model)

    simulator = MujocoSimulator(model=model, data=data, simulation_rate=1/cfg.mujoco.simulation_hz)
    renderer = MujocoRenderer(model=model, data=data, render_resolution=(width, height))
    inverse_kinematics = InverseKinematics(data=data)

    simulator.reset()
    initial_position = simulator.initial_position

    simulator_cs = MujocoSimulatorCS(
        simulator=simulator,
        simulation_rate=1/cfg.mujoco.simulation_hz,
        render_rate=1/cfg.mujoco.observation_hz,
        renderer=renderer,
        inverse_kinematics=inverse_kinematics,
    )

    if cfg.control == 'gui':
        episode_metadata = {
            'mujoco_model_path': cfg.mujoco.model_path,
            'simulation_hz': cfg.mujoco.simulation_hz,
        }
        control = DearpyguiUi(width, height, episode_metadata, initial_position)
    else:
        control = WebXR(port=cfg.webxr.port)

    return simulator_cs, control

async def setup_robot(cfg: DictConfig):
    """Setup and return physical robot components"""
    franka = Franka(cfg.franka.ip, cfg.franka.relative_dynamics_factor, cfg.franka.gripper_force)

    if cfg.control == 'gui':
        # Create GUI control for physical robot
        initial_position = Transform3D(
            translation=franka.robot.current_pose.end_effector_pose.translation,
            quaternion=franka.robot.current_pose.end_effector_pose.quaternion
        )
        control = DearpyguiUi(
            width=cfg.camera.width if hasattr(cfg.camera, 'width') else 320,
            height=cfg.camera.height if hasattr(cfg.camera, 'height') else 240,
            initial_position=initial_position
        )
    else:
        control = WebXR(port=cfg.webxr.port)

    gripper = None
    if 'dh_gripper' in cfg:
        gripper = DHGripper(cfg.dh_gripper)

    return franka, gripper, control

async def main_async(cfg: DictConfig):
    components: List[ir.ControlSystem] = []

    # Setup robot/simulator and control method
    if cfg.robot == 'mujoco':
        simulator, control = await setup_mujoco(cfg)
        components.extend([simulator, control])

        # Connect control to simulator
        simulator.bind(
            gripper_target_grasp=control.outs.gripper_target_grasp,
            robot_target_position=control.outs.robot_target_position,
            reset=control.outs.reset,
        )
        control.bind(
            images=simulator.outs.images,
            robot_position=simulator.outs.robot_position,
        )

        robot_data_source = simulator

    else:  # Physical robot
        franka, gripper, control = await setup_robot(cfg)
        components.extend([franka, control])

        # Connect control to robot
        franka.bind(target_position=control.outs.robot_target_position)
        control.bind(robot_position=franka.outs.position)

        if gripper:
            gripper.bind(grip=control.outs.gripper_target_grasp)
            components.append(gripper)

        robot_data_source = franka

    # Setup camera if needed
    if cfg.robot == 'physical' or cfg.extra_camera:
        cam = hardware.from_config.sl_camera(cfg.camera)
        components.append(cam)
        image_source = cam.outs.frame
    else:
        image_source = simulator.outs.images

    # Setup data collection if enabled
    if cfg.data_output_dir is not None:
        properties_to_dump = ir.utils.properties_dict(
            robot_joints=robot_data_source.outs.joint_positions,
            robot_position_translation=ir.utils.map_property(lambda t: t.translation, robot_data_source.outs.position),
            robot_position_quaternion=ir.utils.map_property(lambda t: t.quaternion, robot_data_source.outs.position),
            ext_force_ee=robot_data_source.outs.ext_force_ee,
            ext_force_base=robot_data_source.outs.ext_force_base,
            grip=gripper.outs.grip if gripper else None
        )

        data_dumper = DatasetDumper(cfg.data_output_dir)
        components.append(
            data_dumper.bind(
                image=image_source,
                target_grip=control.outs.gripper_target_grasp,
                target_robot_position=control.outs.robot_target_position,
                start_episode=control.outs.start_tracking,
                end_episode=control.outs.stop_tracking,
                robot_data=properties_to_dump,
            )
        )

    # Run the system
    system = ir.compose(*components)
    await ir.utils.run_gracefully(system)

@hydra.main(version_base=None, config_path="configs", config_name="data_collection")
def main(cfg: DictConfig):
    asyncio.run(main_async(cfg))

if __name__ == "__main__":
    main()
