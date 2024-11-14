from logging.config import dictConfig
from pathlib import Path
from tqdm import tqdm
from control import utils
from control.utils import PropDict, control_system_fn
from geom import Transform3D
import numpy as np
import hydra
import mujoco
import torch

from control.system import ControlSystem, control_system
from control.world import MainThreadWorld, World
from simulator.mujoco.environment import MujocoRenderer, MujocoSimulator
from tools.dataset_dumper import DatasetDumper


@control_system(outputs=["actuator_values", "target_grip", "target_robot_position", "start_episode", "end_episode"])
class RecordPlayer(ControlSystem):
    def __init__(self, world: World, episode_path: Path):
        super().__init__(world)
        self.episode_path = Path(episode_path)
        
    def run(self):
        data = torch.load(self.episode_path)
        n_frames = len(data['time'])
        
        self.outs.start_episode.write(True, data['time/robot'][0])

        for i in tqdm(range(n_frames)):
            simulator_ts = data['time/robot'][i]
            self.outs.actuator_values.write(data['actuator_values'][i], simulator_ts)
            self.outs.target_grip.write(data['target_grip'][i], simulator_ts)

            translation = torch.tensor(data['target_robot_position.translation'][i])
            quaternion = torch.tensor(data['target_robot_position.quaternion'][i])
            
            self.outs.target_robot_position.write(Transform3D(translation, quaternion), simulator_ts)

        self.outs.end_episode.write(True, data['time/robot'][-1])



@hydra.main(version_base=None, config_path=".", config_name="record_renderer")
def main(cfg: dictConfig):
    world = MainThreadWorld()
    model = mujoco.MjModel.from_xml_path(cfg.mujoco.model_path)
    data = mujoco.MjData(model)
    record_player = RecordPlayer(world, Path(cfg.episode_path))
    simulator = MujocoSimulator(world, model, data)
    renderer = MujocoRenderer(
        world, model, data, render_resolution=(cfg.mujoco.camera_width, cfg.mujoco.camera_height), max_fps=cfg.mujoco.observation_hz
    )
    dumper = DatasetDumper(world, cfg.data_output_dir)
    simulator.ins.bind(actuator_values=record_player.outs.actuator_values,
                       target_grip=record_player.outs.target_grip)
    renderer.ins.bind(step_complete=simulator.outs.step_complete)
    
    properties_to_dump = PropDict(world, {
        'robot_joints': simulator.outs.joints,
        'robot_position.translation': simulator.outs.robot_translation,
        'robot_position.quaternion': simulator.outs.robot_quaternion,
        'ext_force_ee': simulator.outs.ext_force_ee,
        'ext_force_base': simulator.outs.ext_force_base,
        'grip': simulator.outs.grip,
        'actuator_values': simulator.outs.actuator_values,
    })

    @utils.map_port
    def stack_images(images):
        return np.hstack([images['handcam_left'], images['handcam_right']])

    dumper.ins.bind(
        start_episode=record_player.outs.start_episode,
        end_episode=record_player.outs.end_episode,
        target_grip=record_player.outs.target_grip,
        target_robot_position=record_player.outs.target_robot_position,
        image=stack_images(renderer.outs.images),
        robot_data=properties_to_dump.outs.prop_values,
    )

    @control_system_fn(inputs=['episode_saved'])
    def on_episode_saved(ins, outs):
        for _ts, _value in ins.episode_saved.read_until_stop():
            world.stop_event.set()

    stopper = on_episode_saved(world)
    stopper.ins.bind(episode_saved=dumper.outs.episode_saved)

    world.run()

if __name__ == "__main__":
    main()