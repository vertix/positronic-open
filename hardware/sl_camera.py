# Control systems for StereoLabs cameras

from dataclasses import dataclass
from queue import Full, Empty
from typing import Optional

import pyzed.sl as sl
import multiprocessing as mp
from multiprocessing import Queue
import numpy as np

from control import ControlSystem, utils, World, MainThreadWorld, control_system
from tools.video import VideoDumper

SlImage = sl.Mat

@dataclass
class Record:
    success: bool
    image: Optional[SlImage] = None


@control_system(outputs=['record'])
class SLCamera(ControlSystem):
    def __init__(self, world: World, fps=30, view=sl.VIEW.LEFT, resolution=sl.RESOLUTION.AUTO):
        super().__init__(world)
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = resolution
        self.init_params.camera_fps = fps
        self.init_params.depth_mode = sl.DEPTH_MODE.NONE
        self.init_params.sdk_verbose = 1
        self.init_params.enable_image_enhancement = False
        self.init_params.async_grab_camera_recovery = False
        self.view = view
        self.frame_queue = Queue(maxsize=5)  # Limit queue size to prevent memory issues
        self.process = None

    def run(self):
        self.process = mp.Process(target=self._camera_process, args=(self.frame_queue,))
        self.process.start()

        while not self.should_stop:
            try:
                record, ts_ms = self.frame_queue.get(timeout=1)
                self.outs.record.write(record, ts_ms)
            except Empty:
                continue

        if self.process.is_alive():
            self.process.terminate()
            self.process.join()

    def _camera_process(self, queue: Queue):
        zed = sl.Camera()
        zed.open(self.init_params)
        SUCCESS = sl.ERROR_CODE.SUCCESS
        TIME_REF_IMAGE = sl.TIME_REFERENCE.IMAGE

        try:
            fps = utils.FPSCounter("Camera")
            while True:
                result = zed.grab()
                fps.tick()
                if result != SUCCESS:
                    queue.put((Record(success=False), None))
                    continue

                image = sl.Mat()
                if zed.retrieve_image(image, self.view) != SUCCESS:
                    queue.put((Record(success=False), None))
                    continue

                ts_ms = zed.get_timestamp(TIME_REF_IMAGE).get_milliseconds()
                # Convert image to numpy array to make it picklable
                np_image = image.get_data()[:, :, [2, 1, 0]]
                queue.put((Record(success=True, image=np_image), ts_ms), block=True)
        except Full:
            pass  # Skip frame if queue is full
        finally:
            zed.close()


# Test SLCamera system
def main():
    world = MainThreadWorld()
    camera = SLCamera(world, fps=15, view=sl.VIEW.SIDE_BY_SIDE, resolution=sl.RESOLUTION.VGA)
    video_dumper = VideoDumper(world, "video.mp4", 15, codec='libx264')

    @utils.map_port
    def extract_np_image(record):
        if record is None or not record.success:
            return None
        return record.image

    video_dumper.ins.image = extract_np_image(camera.outs.record)

    world.run()


if __name__ == "__main__":
    main()
