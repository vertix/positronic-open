from collections import defaultdict
import os

import numpy as np
import torch


import ironic as ir

class SerialDumper:
    def __init__(self, directory: str):
        """
        Dumps serial data to a directory as torch tensors.

        Args:
            directory: Directory to save the data to. Episodes will be named sequentially.

        Example:
            >>> dumper = SerialDumper("data")
            >>> dumper.start_episode()
            >>> dumper.write({"position": np.array([1, 2, 3]), "time": 0.1})
            >>> dumper.write({"position": np.array([4, 5, 6]), "time": 0.2})
            >>> dumper.end_episode(metadata={"robot_type": "franka"})
        """
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

    def end_episode(self, metadata: dict = None):
        self.dump_episode(metadata=metadata)
        self.data = defaultdict(list)
        self.episode_metadata = {}

    def write(self, data: dict):
        for k, v in data.items():
            self.data[k].append(v)

    def dump_episode(self, metadata: dict = None):
        # Transform everything to torch tensors
        for k, v in self.data.items():
            self.data[k] = torch.tensor(np.array(v))

        if metadata is not None:
            self.data.update(metadata)

        fname = f"{self.directory}/{str(self.episode_count).zfill(3)}.pt"
        torch.save(dict(self.data), fname)
        print(f"Episode {self.episode_count} saved to {fname} with {len(self.data['time'])} frames")


@ir.ironic_system(
    input_ports=['image', 'start_episode', 'end_episode', 'target_grip', 'target_robot_position'],
    input_props=['robot_data'])
class DatasetDumper(ir.ControlSystem):
    def __init__(self,  directory: str):
        super().__init__()
        self.dumper = SerialDumper(directory)

        self.tracked = False
        self.episode_start = None
        self.target_grip, self.target_robot_position, self.target_ts = None, None, None

    def dump_episode(self):
        self.dumper.dump_episode()

    @ir.on_message('start_episode')
    async def on_start_episode(self, _message: ir.Message):
        self.tracked = True
        self.episode_start = ir.system_clock()
        print(f"Episode {self.dumper.episode_count} started")
        self.dumper.start_episode()

    @ir.on_message('end_episode')
    async def on_end_episode(self, message: ir.Message):
        assert self.tracked, "end_episode without start_episode"
        self.tracked = False
        self.dumper.end_episode(metadata=message.data)
        print(f"Episode {self.dumper.episode_count} ended")

    @ir.on_message('target_grip')
    async def on_target_grip(self, message: ir.Message):
        self.target_grip, self.target_ts = message.data, message.timestamp

    @ir.on_message('target_robot_position')
    async def on_target_robot_position(self, message: ir.Message):
        self.target_robot_position = message.data

    @ir.on_message('image')
    async def on_image(self, image_message: ir.Message):
        if not self.tracked:
            return

        if self.target_robot_position is None:
            print("No target robot position")
            return

        ep_dict = {}
        for key, image in image_message.data.items():
            name = f'image_{key}' if key else 'image'
            ep_dict[name] = image
        ep_dict['target_grip'] = self.target_grip
        ep_dict['target_robot_position_translation'] = self.target_robot_position.translation
        ep_dict['target_robot_position_quaternion'] = self.target_robot_position.quaternion

        robot_message = await self.ins.robot_data()
        for name, value in robot_message.data.items():
            ep_dict[name] = value

        # HACK: Here we use knowledge that time is in nanoseconds
        now_ts = ir.system_clock()
        ep_dict['time'] = (now_ts - self.episode_start) / 1e9
        ep_dict['delay/image'] = (now_ts - image_message.timestamp) / 1e9
        ep_dict['delay/robot'] = (now_ts - robot_message.timestamp) / 1e9
        ep_dict['delay/target'] = (now_ts - self.target_ts) / 1e9

        self.dumper.write(ep_dict)
