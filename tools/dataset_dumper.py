from collections import defaultdict
import os

import numpy as np
import torch

from control import ControlSystem, World, control_system


class SerialDumper:
    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)
        self.data = defaultdict(list)
        self.episode_metadata = {}

        episode_files = [f for f in os.listdir(directory) if f.endswith('.pt')]
        if episode_files:
            episode_numbers = [int(f.split('.')[0]) for f in episode_files]
            self.episode_count = max(episode_numbers) + 1
        else:
            self.episode_count = 0

    def start_episode(self):
        self.episode_count += 1

    def end_episode(self):
        self.dump_episode()
        self.data = defaultdict(list)
        self.episode_metadata = {}
        
    def write(self, data: dict):
        for k, v in data.items():
            self.data[k].append(v)

    def add_metadata(self, metadata: dict):
        self.episode_metadata.update(metadata)

    def dump_episode(self):
        # Transform everything to torch tensors
        for k, v in self.data.items():
            self.data[k] = torch.tensor(np.array(v))
        
        self.data.update(self.episode_metadata)

        fname = f"{self.directory}/{str(self.episode_count).zfill(3)}.pt"
        torch.save(dict(self.data), fname)
        print(f"Episode {self.episode_count} saved to {fname} with {len(self.data['time'])} frames")


@control_system(inputs=['image', 'start_episode', 'end_episode', 'target_grip', 'target_robot_position', 'metadata'],
                outputs=['episode_saved'],
                input_props=['robot_data'])
class DatasetDumper(ControlSystem):
    def __init__(self, world: World, directory: str):
        super().__init__(world)
        self.dumper = SerialDumper(directory)

    def dump_episode(self):
        self.dumper.dump_episode()
        self.outs.episode_saved.write(True, self.world.now_ts)

    def run(self):
        tracked = False
        episode_start = None
        target_grip, target_robot_position, img, target_ts = None, None, None, None

        for name, ts, data in self.ins.read():
            if name == 'start_episode':
                tracked = True
                print(f"Episode {self.dumper.episode_count} started")
                episode_start = self.world.now_ts
                self.dumper.start_episode()
            elif name == 'end_episode':
                assert tracked, "end_episode without start_episode"
                self.dumper.end_episode()
                self.outs.episode_saved.write(True, self.world.now_ts)
                episode_start = None
                tracked = False
            elif tracked and name == 'target_grip':
                target_grip, target_ts = data, ts
            elif tracked and name == 'target_robot_position':
                target_robot_position = data
            elif name == 'metadata':
                assert tracked, "Metadata added, but no episode started"
                self.dumper.add_metadata(data)
            elif tracked and name == 'image' and target_ts is not None:
                ep_dict = {}
                now_ts = self.world.now_ts

                img = data.image if hasattr(data, 'image') else data
                ep_dict['image'] = img
                ep_dict['target_grip'] = target_grip
                ep_dict['target_robot_position.translation'] = target_robot_position.translation
                ep_dict['target_robot_position.quaternion'] = target_robot_position.quaternion

                data, robot_ts = self.ins.robot_data()
                for name, value in data.items():
                    ep_dict[name] = value
                
                ep_dict['time'] = (now_ts - episode_start) / 1000
                ep_dict['time/robot'] = robot_ts
                ep_dict['delay/image'] = (now_ts - ts) / 1000
                ep_dict['delay/robot'] = (now_ts - robot_ts) / 1000
                ep_dict['delay/target'] = (now_ts - target_ts) / 1000

                self.dumper.write(ep_dict)
