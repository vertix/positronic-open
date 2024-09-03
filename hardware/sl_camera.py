# Control systems for StereoLabs cameras

import asyncio
from dataclasses import dataclass
from typing import Optional
import threading
from queue import Queue

import cv2
import pyzed.sl as sl

from control import ControlSystem

SlImage = sl.Mat

@dataclass
class Record:
    success: bool
    image: Optional[SlImage] = None


class SLCamera(ControlSystem):
    def __init__(self, fps=30, view=sl.VIEW.LEFT, resolution=sl.RESOLUTION.AUTO):
        super().__init__(inputs=[], outputs=['record'])

        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = resolution
        self.init_params.camera_fps = fps
        self.init_params.depth_mode = sl.DEPTH_MODE.NONE
        self.init_params.sdk_verbose = 1

        self.view = view
        self.queue = Queue()
        self.stop_event = threading.Event()

    def read_camera_data(self):
        zed = sl.Camera()
        zed.open(self.init_params)
        try:
            while not self.stop_event.is_set():
                result = zed.grab()
                if result != sl.ERROR_CODE.SUCCESS:
                    # TODO: Should we be more specific about the error?
                    # See(https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1ERROR__CODE.html)
                    self.queue.put((False, None, None))
                    continue

                image = sl.Mat()
                if zed.retrieve_image(image, self.view) != sl.ERROR_CODE.SUCCESS:
                    self.queue.put((False, None, None))
                    continue

                ts_ms = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
                self.queue.put((True, image, ts_ms))
        finally:
            zed.close()

    async def run(self):
        thread = threading.Thread(target=self.read_camera_data)
        thread.start()
        try:
            while True:
                success, image, ts_ms = await asyncio.get_event_loop().run_in_executor(None, self.queue.get)
                if not success:
                    await self.outs.record.write(Record(success=False))
                else:
                    await self.outs.record.write(Record(success=True, image=image), timestamp=ts_ms)
        finally:
            self.stop_event.set()
            thread.join()


# Test SLCamera system
async def _main():
    class VideoDumper(ControlSystem):
        def __init__(self, filename: str, fps: int):
            super().__init__(inputs=['record'], outputs=[])
            self.filename = filename
            self.fps = fps

        async def run(self):
            video_writer = None
            try:
                while True:
                    _, record = await self.ins.record.read()
                    if record is None or not record.success:
                        continue

                    image = record.image
                    if video_writer is None:
                        video_writer = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*'XVID'),
                                                       self.fps, (image.get_width(), image.get_height()))

                    video_writer.write(cv2.cvtColor(image.get_data()[:, :, :3], cv2.COLOR_BGR2RGB))
            finally:
                if video_writer is not None:
                    video_writer.release()


    camera = SLCamera(fps=30, view=sl.VIEW.SIDE_BY_SIDE)
    video_dumper = VideoDumper("video.avi", 30)
    video_dumper.ins.record = camera.outs.record

    await asyncio.gather(video_dumper.run(), camera.run())

if __name__ == "__main__":
    asyncio.run(_main())
