import asyncio
import logging
from typing import List

import hydra
from omegaconf import DictConfig

import ironic as ir
from hardware import from_config
from tools.dataset_dumper import DatasetDumper

logging.basicConfig(level=logging.INFO,
                    handlers=[logging.StreamHandler()])


def setup_interface(cfg: DictConfig):
    if cfg.type == 'teleop':
        from webxr import WebXR
        from teleop import TeleopSystem

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
        outputs['start_recording'] = (teleop, 'start_recording')
        outputs['stop_recording'] = (teleop, 'stop_recording')
        outputs['reset'] = (teleop, 'reset')
        return ir.compose(*components, inputs=inputs, outputs=outputs), {}
    elif cfg.type == 'gui':
        from simulator.mujoco.mujoco_gui import DearpyguiUi

        return DearpyguiUi(cfg.mujoco.camera_names), {}
    else:
        raise ValueError(f"Invalid control type: {cfg.type}")


async def main_async(cfg: DictConfig):
    metadata = {}
    control, md = setup_interface(cfg.control_ui)
    metadata.update(md)

    hardware, md = from_config.robot_setup(cfg.hardware)
    metadata.update(md)

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

    if cfg.get('rerun'):
        from tools.rerun_vis import RerunVisualiser

        visualizer = RerunVisualiser()
        visualizer.bind(
            frame=hardware.outs.frame,
            ext_force_ee=hardware.outs.ext_force_ee,
            ext_force_base=hardware.outs.ext_force_base,
            robot_position=hardware.outs.robot_position
        )
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
                start_episode=control.outs.start_recording,
                end_episode=control.outs.stop_recording,
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
