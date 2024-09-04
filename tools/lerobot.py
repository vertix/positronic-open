from collections import defaultdict
import time

from control import ControlSystem, World

class LerobotDatasetDumper(ControlSystem):
    def __init__(self, world: World, filename: str, video_filename: str):
        super().__init__(world, inputs=['image', 'ext_force_ee', 'ext_force_base', 'robot_position', 'robot_joints',
                                 'start_episode', 'end_episode'], outputs=[])
        self.filename = filename
        self.video_filename = video_filename

    async def run(self):
        # ep_dict = defaultdict(list)
        # last = {}
        start_time = time.time()
        frame_count = 0
        tracked = False

        async for name, ts, data in self.ins.read():
            if name == 'start_episode':
                start_time = time.time()
                frame_count = 0
                tracked = True
            elif name == 'end_episode':
                print(f"Episode ended after {time.time() - start_time} seconds. Average FPS: {frame_count / (time.time() - start_time)}")
            elif tracked and name == 'image':
                frame_count += 1

            # if name != 'image':
            #     last[name] = data
            # else:
            #     frame_count += 1
            #     if frame_count % 30 == 0:
            #         print(f"FPS: {frame_count / (time.time() - start_time)}")
            #     # ep_dict['image'].append(data)
            #     # for name, data in last.items():
            #     #     ep_dict[name].append(data)

