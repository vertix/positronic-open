# Control systems for StereoLabs cameras

import asyncio
from dataclasses import dataclass
from typing import Optional

import cv2
import pyzed.sl as sl

from control import ControlSystem
from control.system import EventSystem


@dataclass
class Record:
    success: bool
    ts_ms: Optional[int] = None
    image: Optional[sl.Mat] = None


class SLCamera(ControlSystem):
    def __init__(self, fps=30, view=sl.VIEW.LEFT):
        super().__init__(inputs=[], outputs=['record'])

        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.AUTO
        self.init_params.camera_fps = fps
        self.init_params.depth_mode = sl.DEPTH_MODE.NONE
        self.init_params.sdk_verbose = 1

        self.view = view

    async def run(self):
        zed = sl.Camera()
        zed.open(self.init_params)
        try:
            while True:
                result = zed.grab()
                if result != sl.ERROR_CODE.SUCCESS:
                    # TODO: Should we be more specific about the error?
                    # See(https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1ERROR__CODE.html)
                    await self.outs.record.write(Record(success=False))
                    continue

                image = sl.Mat()
                if zed.retrieve_image(image, self.view) != sl.ERROR_CODE.SUCCESS:
                    await self.outs.record.write(Record(success=False))
                    continue

                ts_ms = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
                await self.outs.record.write(Record(success=True, ts_ms=ts_ms, image=image))
        finally:
            zed.close()


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
