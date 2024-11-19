# Control systems for StereoLabs cameras

from queue import Full, Empty
import multiprocessing as mp
from multiprocessing import Queue
import asyncio

import numpy as np
import pyzed.sl as sl
import ironic as ir


@ir.ironic_system(output_ports=['frame'])
class SLCamera(ir.ControlSystem):
    def __init__(self, fps=30, view=sl.VIEW.LEFT, resolution=sl.RESOLUTION.AUTO):
        super().__init__()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = resolution
        self.init_params.camera_fps = fps
        self.init_params.depth_mode = sl.DEPTH_MODE.NONE
        self.init_params.sdk_verbose = 1
        self.init_params.enable_image_enhancement = False
        self.init_params.async_grab_camera_recovery = False
        self.view = view
        self.frame_queue = Queue(maxsize=5)
        self.process = None
        self.fps = None

    async def setup(self):
        self.process = mp.Process(target=self._camera_process, args=(self.frame_queue,))
        self.process.start()
        self.fps = ir.utils.FPSCounter("Camera")

    async def cleanup(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()

    async def step(self):
        try:
            image, ts_ms = self.frame_queue.get(timeout=1)
            await self.outs.frame.write(ir.Message(data=image, timestamp=ts_ms))
            self.fps.tick()
        except Empty:
            await asyncio.sleep(1 / self.init_params.camera_fps)

    def _camera_process(self, queue: Queue):
        zed = sl.Camera()
        zed.open(self.init_params)
        SUCCESS = sl.ERROR_CODE.SUCCESS
        TIME_REF_IMAGE = sl.TIME_REFERENCE.IMAGE

        try:
            while True:
                result = zed.grab()
                if result != SUCCESS:
                    queue.put((None, 0))
                    continue

                image = sl.Mat()
                if zed.retrieve_image(image, self.view) != SUCCESS:
                    queue.put((None, 0))
                    continue

                ts_ms = zed.get_timestamp(TIME_REF_IMAGE).get_nanoseconds()
                np_image = image.get_data()[:, :, [0, 1, 2]]
                queue.put((np_image, ts_ms), block=True)
        except Full:
            pass  # Skip frame if queue is full
        finally:
            zed.close()


if __name__ == "__main__":
    import asyncio
    from tools.video import VideoDumper

    async def _main():
        camera = SLCamera()
        system = ir.compose(
            camera,
            VideoDumper("video.mp4", 15, codec='libx264').bind(image=camera.outs.frame)
        )

        await system.setup()

        try:
            while True:
                await system.step()
                await asyncio.sleep(0.0)
        finally:
            print("Cleaning up")
            await system.cleanup()
            print("Done")

    asyncio.run(_main())
