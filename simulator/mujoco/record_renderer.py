from tqdm import tqdm
import numpy as np
import hydra
import mujoco
import torch

from simulator.mujoco.environment import MujocoRenderer
from simulator.mujoco.sim import MujocoSimulator
from tools.dataset_dumper import SerialDumper


@hydra.main(version_base=None, config_path="configs", config_name="record_renderer")
def main(cfg):
    data = torch.load(cfg.episode_path)
    n_frames = len(data['image_timestamp'])

    mj_model = mujoco.MjModel.from_xml_path(data['mujoco_model_path'])
    mj_data = mujoco.MjData(mj_model)
    simulator = MujocoSimulator(mj_model, mj_data, simulation_rate=1 / cfg.mujoco.simulation_hz)
    renderer = MujocoRenderer(mj_model, mj_data, camera_names=cfg.mujoco.camera_names, render_resolution=(cfg.mujoco.camera_width, cfg.mujoco.camera_height))
    dataset_writer = SerialDumper(cfg.data_output_dir)

    simulator.reset()
    renderer.initialize()
    dataset_writer.start_episode()

    event_idx = 0
    last_render_ts = 0
    tqdm_iter = tqdm(total=n_frames)

    while event_idx < n_frames:
        if data['image_timestamp'][event_idx] <= simulator.ts_ns:
            simulator.set_actuator_values(data['actuator_values'][event_idx])
            simulator.set_grip(data['target_grip'][event_idx])
            event_idx += 1
            tqdm_iter.update(1)

        tqdm_iter.set_postfix(sim_ts=simulator.ts_sec)
        simulator.step()

        if simulator.ts_ns - last_render_ts >= 1e9 / cfg.mujoco.observation_hz:
            last_render_ts = simulator.ts_ns
            images = renderer.render_frames()

            dataset_writer.write({
                **{f'image.{mapped_name}': images[orig_name] for mapped_name, orig_name in cfg.image_name_mapping.items()},
                'actuator_values': simulator.actuator_values,
                'grip': simulator.grip,
                'robot_position_translation': simulator.robot_position.translation,
                'robot_position_quaternion': simulator.robot_position.quaternion,
                'ext_force_ee': simulator.ext_force_ee,
                'ext_force_base': simulator.ext_force_base,
                'robot_joints': simulator.joints,
                'target_grip': data['target_grip'][event_idx],
                'target_robot_position_quaternion': data['target_robot_position_quaternion'][event_idx],
                'target_robot_position_translation': data['target_robot_position_translation'][event_idx],
                'image_timestamp': data['image_timestamp'][event_idx],
                'robot_timestamp': data['robot_timestamp'][event_idx],
                'target_timestamp': data['target_timestamp'][event_idx],
            })

    dataset_writer.end_episode()
    tqdm_iter.close()



if __name__ == "__main__":
    main()