# Control systems for StereoLabs cameras

import asyncio
from dataclasses import dataclass
import queue
from typing import Optional
import threading
from queue import Queue

import pyzed.sl as sl

from control import ControlSystem, utils, World, MainThreadWorld, ThreadWorld
from tools.video import VideoDumper

SlImage = sl.Mat

@dataclass
class Record:
    success: bool
    image: Optional[SlImage] = None


class SLCamera(ControlSystem):
    def __init__(self, world: World, fps=30, view=sl.VIEW.LEFT, resolution=sl.RESOLUTION.AUTO):
        super().__init__(world, inputs=[], outputs=['record'])

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
        SUCCESS = sl.ERROR_CODE.SUCCESS
        TIME_REF_IMAGE = sl.TIME_REFERENCE.IMAGE

        try:
            while not self.stop_event.is_set():
                result = zed.grab()
                if result != SUCCESS:
                    # TODO: Should we be more specific about the error?
                    # See(https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1ERROR__CODE.html)
                    self.queue.put((False, None, None))
                    continue

                image = sl.Mat()
                if zed.retrieve_image(image, self.view) != SUCCESS:
                    self.queue.put((False, None, None))
                    continue

                ts_ms = zed.get_timestamp(TIME_REF_IMAGE).get_milliseconds()
                self.queue.put((True, image, ts_ms))
        finally:
            zed.close()

    async def run(self):
        thread = threading.Thread(target=self.read_camera_data)
        thread.start()
        try:
            fps = utils.FPSCounter("Camera")
            while True:
                try:
                    success, image, ts_ms = self.queue.get_nowait()
                    if not success:
                        await self.outs.record.write(Record(success=False))
                    else:
                        await self.outs.record.write(Record(success=True, image=image), timestamp=ts_ms)
                except queue.Empty:
                    await asyncio.sleep(1 / 60)
                    continue
                fps.tick()
        except Exception as e:
            print(f"SLCamera error: {e}")
        finally:
            print("Cancelling SLCamera")
            self.stop_event.set()
            thread.join()
            print("SLCamera cancelled")


# Test SLCamera system
async def _main():
    world = ThreadWorld()
    camera = SLCamera(world, fps=15, view=sl.VIEW.SIDE_BY_SIDE, resolution=sl.RESOLUTION.VGA)
    video_dumper = VideoDumper(world, "video.mp4", 15, codec='libx264')

    @utils.map_port
    def extract_np_image(record):
        if record is None or not record.success:
            return None
        return record.image.get_data()[:, :, :3]

    video_dumper.ins.image = extract_np_image(camera.outs.record)

    await world.run()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("Program interrupted by user, exiting...")
