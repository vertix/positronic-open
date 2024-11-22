from typing import Dict

from omegaconf import DictConfig

import ironic as ir

def add_image_mapping(mapping: Dict[str, str], camera: ir.ControlSystem):
    def map_images(frame):
        return {new_k: frame[k] for k, new_k in mapping.items()}

    map_system = ir.utils.MapControlSystem(map_images)
    map_system.bind(input=camera.outs.frame)
    return ir.compose(camera, map_system, outputs={'frame': (map_system, 'output')})

def sl_camera(cfg: DictConfig):
    from hardware.camera.sl import SLCamera
    import pyzed.sl as sl

    view = getattr(sl.VIEW, cfg.view)
    resolution = getattr(sl.RESOLUTION, cfg.resolution)
    kwargs = {}
    if 'depth_mode' in cfg:
        kwargs['depth_mode'] = getattr(sl.DEPTH_MODE, cfg.depth_mode)
    if 'coordinate_units' in cfg:
        kwargs['coordinate_units'] = getattr(sl.UNIT, cfg.coordinate_units)
    if 'max_depth' in cfg:
        kwargs['max_depth'] = cfg.max_depth
    if 'depth_mask' in cfg:
        kwargs['depth_mask'] = cfg.depth_mask

    camera = SLCamera(cfg.fps, view, resolution, **kwargs)

    if 'image_mapping' in cfg:
        return add_image_mapping(cfg.image_mapping, camera)
    else:
        return camera


def realsense_camera(cfg: DictConfig):
    from hardware.camera.realsense import RealsenseCamera, RealsenseCameraCS

    camera = RealsenseCameraCS(RealsenseCamera(
        resolution=cfg.resolution,
        fps=cfg.fps,
        enable_color=cfg.enable_color,
        enable_depth=cfg.enable_depth,
        enable_infrared=cfg.enable_infrared,
    ))

    if 'image_mapping' in cfg:
        return add_image_mapping(cfg.image_mapping, camera)
    else:
        return camera


def luxonis_camera(cfg: DictConfig):
    from hardware.camera.luxonis import LuxonisCamera, LuxonisCameraCS

    camera = LuxonisCameraCS(LuxonisCamera())

    return camera


def get_camera(cfg: DictConfig):
    if cfg.type == 'sl':
        return sl_camera(cfg)
    elif cfg.type == 'realsense':
        return realsense_camera(cfg)
    elif cfg.type == 'luxonis':
        return luxonis_camera(cfg)
    else:
        raise ValueError(f"Invalid camera type: {cfg.type}")


def robot_setup(cfg: DictConfig):
    """Setup and returns control system with robot and camera(s).
    inputs:
        target_position: Target position for the robot
        target_grip: Target grip for the gripper
    outputs:
        robot_position: Current position of the robot
        grip: Current gripper state
        frame: Current camera frame

    Returns:
        control system, metadata (as a dict)
    """
    components, inputs, outputs = [], {}, {}
    if cfg.type == 'physical':
        from hardware import franka
        kwargs = {}
        if 'collision_behavior' in cfg.franka:
            kwargs['collision_behavior'] = cfg.franka.collision_behavior
        if 'home_joints_config' in cfg.franka:
            kwargs['home_joints_config'] = cfg.franka.home_joints_config
        if 'cartesian_mode' in cfg.franka:
            kwargs['cartesian_mode'] = getattr(franka.CartesianMode, cfg.franka.cartesian_mode)

        franka = franka.Franka(cfg.franka.ip, cfg.franka.relative_dynamics_factor, cfg.franka.gripper_force, **kwargs)
        components.append(franka)
        outputs['robot_position'] = (franka, 'position')
        outputs['joint_positions'] = (franka, 'joint_positions')
        outputs['ext_force_base'] = (franka, 'ext_force_base')
        outputs['ext_force_ee'] = (franka, 'ext_force_ee')
        outputs['robot_state'] = (franka, 'state')
        inputs['target_position'] = (franka, 'target_position')
        inputs['reset'] = (franka, 'reset')

        if 'dh_gripper' in cfg:
            from hardware.dhgrp import DHGripper
            gripper = DHGripper(cfg.dh_gripper)
            components.append(gripper)
            outputs['grip'] = (gripper, 'grip')
            inputs['target_grip'] = (gripper, 'target_grip')
        else:
            outputs['grip'] = (franka, 'grip')
            inputs['target_grip'] = (franka, 'target_grip')

        if 'camera' in cfg:
            camera = get_camera(cfg.camera)
            components.append(camera)
            outputs['frame'] = (camera, 'frame')
        else:
            outputs['frame'] = None
        return ir.compose(*components, inputs=inputs, outputs=outputs), {}
    elif cfg.type == 'mujoco':
        from simulator.mujoco.sim import MujocoSimulator, MujocoRenderer, InverseKinematics
        from simulator.mujoco.environment import MujocoSimulatorCS

        simulator = MujocoSimulator.load_from_xml_path(
            model_path=cfg.mujoco.model_path,
            simulation_rate=1/cfg.mujoco.simulation_hz
        )
        renderer = MujocoRenderer(
            model=simulator.model,
            data=simulator.data,
            camera_names=cfg.mujoco.camera_names,
            render_resolution=(cfg.mujoco.camera_width, cfg.mujoco.camera_height)
        )
        inverse_kinematics = InverseKinematics(data=simulator.data)

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

        metadata = {'mujoco_model_path': cfg.mujoco.model_path, 'simulation_hz': cfg.mujoco.simulation_hz}
        return ir.compose(*components, inputs=inputs, outputs=outputs), metadata
    else:
        raise ValueError(f"Invalid robot type: {cfg.type}")
