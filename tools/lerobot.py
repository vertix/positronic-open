from collections import defaultdict
import time

import numpy as np
import torch

from control import ControlSystem, World

class LerobotDatasetDumper(ControlSystem):
    def __init__(self, world: World, directory: str):
        super().__init__(world, inputs=['image', 'ext_force_ee', 'ext_force_base', 'robot_position', 'robot_joints',
                                        'start_episode', 'end_episode'], outputs=[])
        self.directory = directory
        import os

        episode_files = [f for f in os.listdir(directory) if f.endswith('.pt')]
        if episode_files:
            episode_numbers = [int(f.split('.')[0]) for f in episode_files]
            self.episode_count = max(episode_numbers) + 1
        else:
            self.episode_count = 0

    def dump_episode(self, ep_dict):
        # Transform everything to torch tensors
        for k, v in ep_dict.items():
            ep_dict[k] = torch.tensor(np.array(v))
        fname = f"{self.directory}/{self.episode_count}.pt"
        torch.save(ep_dict, fname)
        print(f"Episode {self.episode_count} saved to {fname} with {ep_dict['image'].shape[0]} frames")
        self.episode_count += 1

    def run(self):
        ep_dict = defaultdict(list)

        tracked = False
        episode_start = None

        for name, ts, data in self.ins.read():
            if name == 'start_episode':
                tracked = True
                print(f"Episode {self.episode_count} started")
            elif name == 'end_episode':
                self.dump_episode(ep_dict)
                ep_dict = defaultdict(list)
                episode_start = None
                tracked = False
            elif tracked and name == 'image':
                if None in (self.ins.ext_force_ee.last, self.ins.ext_force_base.last,
                            self.ins.robot_position.last, self.ins.robot_joints.last):
                    continue

                ep_dict['image'].append(data.image[:, :, :3])
                if episode_start is None:
                    episode_start = ts
                ep_dict['time'].append((ts - episode_start) / 1000)
                ep_dict['ee_force'].append(self.ins.ext_force_ee.last[1])
                ep_dict['ee_force'].append(self.ins.ext_force_base.last[1])
                ep_dict['robot_joints'].append(self.ins.robot_joints.last[1])

                _, robot_position = self.ins.robot_position.last
                ep_dict['robot_position_trans'].append(robot_position.translation)
                ep_dict['robot_position_quat'].append(robot_position.quaternion)
