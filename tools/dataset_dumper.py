from collections import defaultdict
import os

import numpy as np
import torch

from control import ControlSystem, World, control_system


@control_system(inputs=['image', 'start_episode', 'end_episode', 'target_grip', 'target_robot_position'],
                input_props=['grip', 'ext_force_ee', 'ext_force_base', 'robot_position', 'robot_joints',])
class DatasetDumper(ControlSystem):
    def __init__(self, world: World, directory: str):
        super().__init__(world)
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

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
        fname = f"{self.directory}/{str(self.episode_count).zfill(3)}.pt"
        torch.save(ep_dict, fname)
        print(f"Episode {self.episode_count} saved to {fname} with {ep_dict['image'].shape[0]} frames")
        self.episode_count += 1

    def run(self):
        ep_dict = defaultdict(list)

        tracked = False
        episode_start = None

        target_grip, target_robot_position, target_ts = None, None, None

        for name, ts, data in self.ins.read():
            if name == 'start_episode':
                tracked = True
                print(f"Episode {self.episode_count} started")
                episode_start = self.world.now_ts
            elif name == 'end_episode':
                assert tracked, "end_episode without start_episode"
                self.dump_episode(ep_dict)
                ep_dict = defaultdict(list)
                episode_start = None
                tracked = False
            elif tracked and name == 'target_grip':
                target_grip, target_ts = data, ts
            elif tracked and name == 'target_robot_position':
                target_robot_position = data
            elif tracked and name == 'image' and target_ts is not None:
                robot_position, robot_ts = self.ins.robot_position()

                now_ts = self.world.now_ts

                img = data.image if hasattr(data, 'image') else data
                ep_dict['image'].append(img)

                ep_dict['target_robot_position.translation'].append(target_robot_position.translation)
                ep_dict['target_robot_position.quaternion'].append(target_robot_position.quaternion)
                ep_dict['target_grip'].append(target_grip)

                if self.ins.grip is not None:
                    ep_dict['grip'].append(self.ins.grip()[0])
                ep_dict['ee_force'].append(self.ins.ext_force_ee()[0])
                ep_dict['base_force'].append(self.ins.ext_force_base()[0])
                ep_dict['robot_joints'].append(self.ins.robot_joints()[0])
                robot_position, robot_ts = self.ins.robot_position()
                ep_dict['robot_position.translation'].append(robot_position.translation)
                ep_dict['robot_position.quaternion'].append(robot_position.quaternion)

                ep_dict['time'].append((now_ts - episode_start) / 1000)
                ep_dict['delay/image'].append((now_ts - ts) / 1000)
                ep_dict['delay/robot'].append((now_ts - robot_ts) / 1000)
                ep_dict['delay/target'].append((now_ts - target_ts) / 1000)
