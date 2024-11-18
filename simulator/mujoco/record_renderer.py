from logging.config import dictConfig
from pathlib import Path
from typing import Tuple

from tqdm import tqdm
import numpy as np
import hydra
import mujoco
import torch

from simulator.mujoco.environment import MujocoRenderer
from simulator.mujoco.sim import MujocoSimulator
from tools.dataset_dumper import SerialDumper


class EpisodeRenderer:
    def __init__(self, render_resolution: Tuple[int, int], simulation_hz: int, observation_hz: int):
        self.render_resolution = render_resolution
        self.simulation_hz = simulation_hz
        self.observation_hz = observation_hz
        
    def render(self, episode_path: Path, target_path: Path):
        data = torch.load(episode_path)
        n_frames = len(data['time'])

        mj_model = mujoco.MjModel.from_xml_path(data['mujoco_model_path'])
        mj_data = mujoco.MjData(mj_model)
        simulator = MujocoSimulator(mj_model, mj_data, simulation_rate=1 / self.simulation_hz)
        renderer = MujocoRenderer(mj_model, mj_data, render_resolution=self.render_resolution)
        dataset_writer = SerialDumper(target_path)

        simulator.reset()
        renderer.initialize()
        dataset_writer.start_episode()

        event_idx = 0
        last_render_ts = 0
        tqdm_iter = tqdm(total=n_frames)

        while event_idx < n_frames:
            if data['time'][event_idx] <= simulator.ts:
                simulator.set_actuator_values(data['actuator_values'][event_idx])
                simulator.set_grip(data['target_grip'][event_idx])
                event_idx += 1
                tqdm_iter.update(1)
            
            tqdm_iter.set_postfix(sim_ts=simulator.ts)
            simulator.step()
            
            if simulator.ts - last_render_ts >= 1 / self.observation_hz:
                last_render_ts = simulator.ts
                images = renderer.render_frames()
                image = np.hstack([images['handcam_left'], images['handcam_right']])
                dataset_writer.write({
                    'image': image,
                    'actuator_values': simulator.actuator_values,
                    'grip': simulator.grip,
                    'robot_position.translation': simulator.robot_position.translation,
                    'robot_position.quaternion': simulator.robot_position.quaternion,
                    'ext_force_ee': simulator.ext_force_ee,
                    'ext_force_base': simulator.ext_force_base,
                    'robot_joints': simulator.joints,
                    'target_grip': data['target_grip'][event_idx],
                    'target_robot_position.quaternion': data['target_robot_position.quaternion'][event_idx],
                    'target_robot_position.translation': data['target_robot_position.translation'][event_idx],
                    'time': data['time'][event_idx],
                    'time/robot': data['time/robot'][event_idx],
                    'delay/robot': data['delay/robot'][event_idx],
                    'delay/target': data['delay/target'][event_idx],
                    'delay/image': data['delay/image'][event_idx],
                })

        dataset_writer.end_episode()
        tqdm_iter.close()


@hydra.main(version_base=None, config_path="configs", config_name="record_renderer")
def main(cfg: dictConfig):
    renderer = EpisodeRenderer(
        (cfg.mujoco.camera_width, cfg.mujoco.camera_height),
        simulation_hz=cfg.mujoco.simulation_hz,
        observation_hz=cfg.mujoco.observation_hz
    )
    renderer.render(cfg.episode_path, cfg.data_output_dir)


if __name__ == "__main__":
    main()