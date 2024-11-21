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
    def __init__(self, fps=30,
                 view=sl.VIEW.LEFT,
                 resolution=sl.RESOLUTION.AUTO,
                 depth_mode=sl.DEPTH_MODE.NONE,
                 coordinate_units=sl.UNIT.METER):
        super().__init__()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = resolution
        self.init_params.camera_fps = fps
        self.init_params.depth_mode = depth_mode
        self.init_params.coordinate_units = coordinate_units
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
            frame, ts_ms = self.frame_queue.get(block=False)
            await self.outs.frame.write(ir.Message(data=frame, timestamp=ts_ms))
            self.fps.tick()
        except Empty:
            await asyncio.sleep(0)

    def _camera_process(self, queue: Queue):
        zed = sl.Camera()
        zed.open(self.init_params)
        SUCCESS = sl.ERROR_CODE.SUCCESS
        TIME_REF_IMAGE = sl.TIME_REFERENCE.IMAGE

        try:
            while True:
                result = zed.grab()
                frame = {}
                if result != SUCCESS:
                    queue.put((frame, 0))
                    continue

                image = sl.Mat()
                ts_ms = zed.get_timestamp(TIME_REF_IMAGE).get_nanoseconds()
                if zed.retrieve_image(image, self.view) == SUCCESS:
                    np_image = image.get_data()[:, :, [0, 1, 2]]

                    if self.view == sl.VIEW.SIDE_BY_SIDE:
                        w = np_image.shape[1] // 2
                        frame['left'] = np_image[:, :w, :3]
                        frame['right'] = np_image[:, w:, :3]
                    elif self.view == sl.VIEW.LEFT:
                        frame['left'] = np_image[:, :, :3]
                    elif self.view == sl.VIEW.RIGHT:
                        frame['right'] = np_image[:, :, :3]
                    else:
                        frame['image'] = np_image[:, :, :3]

                    if self.init_params.depth_mode != sl.DEPTH_MODE.NONE:
                        depth = sl.Mat()
                        if zed.retrieve_measure(depth, sl.MEASURE.DEPTH) == SUCCESS:
                            frame['depth'] = depth.get_data()
                queue.put((frame, ts_ms), block=True)
        except Full:
            pass  # Skip frame if queue is full
        finally:
            zed.close()


if __name__ == "__main__":
    import asyncio
    from tools.video import VideoDumper

    async def _main():
        camera = SLCamera(view=sl.VIEW.LEFT)
        system = ir.compose(
            camera,
            VideoDumper("video.mp4", 30, codec='libx264').bind(
                image=ir.utils.map_port(lambda x: x['left'], camera.outs.frame)
            )
        )

        await ir.utils.run_gracefully(system)

    asyncio.run(_main())
