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

        episode_files = [f for f in os.listdir(directory) if f.endswith('.pt')]
        if episode_files:
            episode_numbers = [int(f.split('.')[0]) for f in episode_files]
            self.episode_count = max(episode_numbers)
        else:
            self.episode_count = 0

    def start_episode(self):
        self.episode_count += 1

    def end_episode(self, metadata: dict = None):
        self.dump_episode(metadata=metadata)
        self.data = defaultdict(list)

    def write(self, data: dict):
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                self.data[k].append(v.copy())
            elif isinstance(v, torch.Tensor):
                self.data[k].append(v.clone())
            elif isinstance(v, list):
                self.data[k].append(v.copy())
            elif isinstance(v, (int, float, str)):
                self.data[k].append(v)
            else:
                print(f"Appending {k} of type {type(v)}. Please check if you need to make a copy.")
                self.data[k].append(v)

    def dump_episode(self, metadata: dict = None):
        # Transform everything to torch tensors
        n_frames, tensor_key = None, None

        for k, v in self.data.items():
            self.data[k] = torch.from_numpy(np.array(v))
            if n_frames is None:
                n_frames = len(self.data[k])
                tensor_key = k
            else:
                # TODO: It's currently assumed, but in the future we might want to support different length tensors.
                assert len(self.data[k]) == n_frames, f"All tensors must have the same length. Got {len(self.data[k])} and {n_frames} for {k} and {tensor_key}."

        if metadata is not None:
            for k in metadata.keys():
                assert k not in self.data, f"Metadata key {k} intersects with data key."

            self.data.update(metadata)

        fname = f"{self.directory}/{str(self.episode_count).zfill(3)}.pt"
        torch.save(dict(self.data), fname)
        print(f"Episode {self.episode_count} saved to {fname} with {n_frames} frames")


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
    async def on_start_episode(self, message: ir.Message):
        self.tracked = True
        self.episode_start = message.timestamp
        print(f"Episode {self.dumper.episode_count} started")
        self.dumper.start_episode()

    @ir.on_message('end_episode')
    async def on_end_episode(self, message: ir.Message):
        assert self.tracked, "end_episode without start_episode"
        self.tracked = False
        metadata = {
            "episode_start": self.episode_start,
            **message.data,
        }
        self.dumper.end_episode(metadata=metadata)
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
            name = f'image.{key}' if key else 'image'
            ep_dict[name] = image
        ep_dict['target_grip'] = self.target_grip
        ep_dict['target_robot_position_translation'] = self.target_robot_position.translation
        ep_dict['target_robot_position_quaternion'] = self.target_robot_position.quaternion

        robot_message = await self.ins.robot_data()
        for name, value in robot_message.data.items():
            ep_dict[name] = value

        ep_dict['image_timestamp'] = image_message.timestamp
        ep_dict['robot_timestamp'] = robot_message.timestamp
        ep_dict['target_timestamp'] = self.target_ts

        self.dumper.write(ep_dict)
