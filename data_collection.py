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
from teleop import TeleopSystem
from tools.dataset_dumper import DatasetDumper
from webxr import WebXR
from simulator.mujoco.mujoco_gui import DearpyguiUi
import mujoco

logging.basicConfig(level=logging.INFO,
                    handlers=[logging.StreamHandler()])


def setup_hardware(cfg: DictConfig):
    """Setup and returns control system with the following interface:
    inputs:
        target_position: Target position for the robot
        target_grip: Target grip for the gripper
    outputs:
        robot_position: Current position of the robot
        grip: Current gripper state
        frame: Current camera frame
    """
    components, inputs, outputs = [], {}, {}
    if cfg.type == 'physical':
        franka = Franka(cfg.franka.ip, cfg.franka.relative_dynamics_factor, cfg.franka.gripper_force)
        components.append(franka)
        outputs['robot_position'] = (franka, 'position')
        outputs['joint_positions'] = (franka, 'joint_positions')
        outputs['ext_force_base'] = (franka, 'ext_force_base')
        outputs['ext_force_ee'] = (franka, 'ext_force_ee')
        outputs['robot_state'] = (franka, 'robot_state')
        inputs['target_position'] = (franka, 'target_position')
        inputs['reset'] = (franka, 'reset')

        if 'dh_gripper' in cfg:
            gripper = DHGripper(cfg.dh_gripper)
            components.append(gripper)
            outputs['grip'] = (gripper, 'grip')
            inputs['target_grip'] = (gripper, 'target_grip')
        else:
            outputs['grip'] = (franka, 'grip')
            inputs['target_grip'] = (franka, 'target_grip')

        camera = hardware.from_config.sl_camera(cfg.camera)
        components.append(camera)
        outputs['frame'] = (camera, 'frame')
    elif cfg.type == 'mujoco':
        model = mujoco.MjModel.from_xml_path(cfg.mujoco.model_path)
        data = mujoco.MjData(model)

        simulator = MujocoSimulator(
            model=model,
            data=data,
            simulation_rate=1/cfg.mujoco.simulation_hz
        )
        renderer = MujocoRenderer(
            model=model,
            data=data,
            camera_names=cfg.mujoco.camera_names,
            render_resolution=(cfg.mujoco.camera_width, cfg.mujoco.camera_height)
        )
        inverse_kinematics = InverseKinematics(data=data)

        simulator.reset()

        # Create MujocoSimulatorCS
        simulator_cs = MujocoSimulatorCS(
            simulator=simulator,
            simulation_rate=1/cfg.mujoco.simulation_hz,
            render_rate=1/cfg.mujoco.observation_hz,
            renderer=renderer,
            inverse_kinematics=inverse_kinematics,
        )
        components.append(simulator_cs)

        # Map the interface
        inputs['target_position'] = (simulator_cs, 'robot_target_position')
        inputs['target_grip'] = (simulator_cs, 'gripper_target_grasp')
        inputs['reset'] = (simulator_cs, 'reset')
        outputs['robot_position'] = (simulator_cs, 'robot_position')
        outputs['joint_positions'] = (simulator_cs, 'actuator_values')
        outputs['ext_force_base'] = (simulator_cs, 'ext_force_base')
        outputs['ext_force_ee'] = (simulator_cs, 'ext_force_ee')
        outputs['grip'] = (simulator_cs, 'grip')
        outputs['frame'] = (simulator_cs, 'images')
        outputs['robot_state'] = (simulator_cs, 'robot_state')
    else:
        raise ValueError(f"Invalid robot type: {cfg.type}")

    return ir.compose(*components, inputs=inputs, outputs=outputs)


def setup_interface(cfg: DictConfig):
    if cfg.control_ui == 'teleop':
        components, inputs, outputs = [], {}, {}
        webxr = WebXR(port=cfg.webxr.port)
        components.append(webxr)

        teleop = TeleopSystem()
        components.append(teleop)

        teleop.bind(
            teleop_transform=webxr.outs.transform,
            teleop_buttons=webxr.outs.buttons,
        )

        inputs['robot_position'] = (teleop, 'robot_position')
        inputs['robot_grip'] = None
        inputs['images'] = None
        inputs['robot_state'] = None
        outputs['robot_target_position'] = (teleop, 'robot_target_position')
        outputs['gripper_target_grasp'] = (teleop, 'gripper_target_grasp')
        outputs['start_tracking'] = (teleop, 'start_tracking')
        outputs['stop_tracking'] = (teleop, 'stop_tracking')
        outputs['reset'] = None  # This ports exists, but not used
        return ir.compose(*components, inputs=inputs, outputs=outputs)
    elif cfg.control_ui == 'gui':
        width, height = cfg.mujoco.camera_width, cfg.mujoco.camera_height

        episode_metadata = {
            'mujoco_model_path': cfg.mujoco.model_path,
            'simulation_hz': cfg.mujoco.simulation_hz,
        }
        # This system has all necessary ports
        return DearpyguiUi(width, height, cfg.mujoco.camera_names, episode_metadata)
    else:
        raise ValueError(f"Invalid control type: {cfg.control_ui}")


async def main_async(cfg: DictConfig):
    control = setup_interface(cfg.control_ui)
    hardware = setup_hardware(cfg.hardware)

    control.bind(
        robot_grip=hardware.outs.grip,
        robot_position=hardware.outs.robot_position,
        images=hardware.outs.frame,
        robot_state=hardware.outs.robot_state,
    )
    hardware.bind(
        target_position=control.outs.robot_target_position,
        target_grip=control.outs.gripper_target_grasp,
        reset=control.outs.reset,
    )

    components: List[ir.ControlSystem] = []
    components.append(control)
    components.append(hardware)

    # Setup data collection if enabled
    if cfg.data_output_dir is not None:
        properties_to_dump = ir.utils.properties_dict(
            robot_joints=hardware.outs.joint_positions,
            robot_position_translation=ir.utils.map_property(lambda t: t.translation, hardware.outs.robot_position),
            robot_position_quaternion=ir.utils.map_property(lambda t: t.quaternion, hardware.outs.robot_position),
            ext_force_ee=hardware.outs.ext_force_ee,
            ext_force_base=hardware.outs.ext_force_base,
            grip=hardware.outs.grip if hardware.outs.grip else None
        )

        data_dumper = DatasetDumper(cfg.data_output_dir)
        components.append(
            data_dumper.bind(
                # TODO: Let user disable images, like in mujoco_gui
                image=hardware.outs.frame,
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
