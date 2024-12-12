from typing import Dict, Tuple

import numpy as np
import pyrealsense2 as rs

from hardware.camera import Camera


class RealsenseCamera(Camera):
    def __init__(self, resolution: Tuple[int, int] = (640, 480), fps: int = 30):
        self.resolution = resolution
        self.fps = fps
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.resolution[0], self.resolution[1], rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.resolution[0], self.resolution[1], rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.infrared, 1, self.resolution[0], self.resolution[1], rs.format.y8, self.fps)
        self.config.enable_stream(rs.stream.infrared, 2, self.resolution[0], self.resolution[1], rs.format.y8, self.fps)


    def setup(self):
        self.pipeline = rs.pipeline()
        self.pipeline.start(self.config)

    def cleanup(self):
        pass

    def get_frame(self) -> Tuple[Dict[str, np.ndarray], float]:
        frames = self.pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        infrared_frame_1 = frames.get_infrared_frame(1)
        infrared_frame_2 = frames.get_infrared_frame(2)
        timestamp = frames.get_timestamp()

        return {
            'image': color_frame.get_data(),
            'depth': depth_frame.get_data(),
            'infrared_1': infrared_frame_1.get_data(),
            'infrared_2': infrared_frame_2.get_data(),
        }, timestamp
