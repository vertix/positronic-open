import asyncio
from dataclasses import dataclass, field, MISSING
import logging
from typing import List

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from hardware import from_config

import ironic as ir
from teleop import TeleopSystem
from tools.dataset_dumper import DatasetDumper
from webxr import WebXR
from simulator.mujoco.mujoco_gui import DearpyguiUi
from tools.rerun_vis import RerunVisualiser

logging.basicConfig(level=logging.INFO,
                    handlers=[logging.StreamHandler()])


def setup_interface(cfg: DictConfig):
    if cfg.type == 'teleop':
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
        outputs['reset'] = (teleop, 'stop_tracking')  # Reset robot when stop tracking
        return ir.compose(*components, inputs=inputs, outputs=outputs), {}
    elif cfg.type == 'gui':
        return DearpyguiUi(cfg.mujoco.camera_names), {}
    else:
        raise ValueError(f"Invalid control type: {cfg.type}")


async def main_async(cfg: DictConfig):
    metadata = {}
    control, md = setup_interface(cfg.control_ui)
    metadata.update(md)

    hardware, md = from_config.robot_setup(cfg.hardware)
    metadata.update(md)

    # Add visualizer
    visualizer = RerunVisualiser()
    visualizer.bind(
        frame=hardware.outs.frame,
        ext_force_ee=hardware.outs.ext_force_ee,
        ext_force_base=hardware.outs.ext_force_base,
        robot_position=hardware.outs.robot_position
    )

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
    components.append(visualizer)  # Add visualizer to components

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

        data_dumper = DatasetDumper(cfg.data_output_dir, additional_metadata=metadata)
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
