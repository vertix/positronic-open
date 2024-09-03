# Control systems for StereoLabs cameras

import asyncio
from dataclasses import dataclass
import queue
import time
from typing import Optional
import threading
from queue import Queue

import av
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
                try:
                    success, image, ts_ms = self.queue.get_nowait()
                    if not success:
                        await self.outs.record.write(Record(success=False))
                    else:
                        await self.outs.record.write(Record(success=True, image=image), timestamp=ts_ms)
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue
        finally:
            self.stop_event.set()
            thread.join()

class VideoDumper(ControlSystem):
    def __init__(self, filename: str, fps: int, width: int = None, height: int = None, codec: str = 'libx264'):
        super().__init__(inputs=['record'], outputs=[])
        self.filename = filename
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec
        self.queue = queue.Queue(maxsize=fps * 5)
        self.stop_event = threading.Event()

    def encode_video(self):
        container = av.open(self.filename, mode='w', format='mp4')
        stream = container.add_stream(self.codec, rate=self.fps)
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '27', 'g': '2', 'preset': 'ultrafast', 'tune': 'zerolatency'}
        first_frame = True

        try:
            while not self.stop_event.is_set():
                record = self.queue.get()
                if record is None or not record.success:
                    continue

                image = record.image
                if first_frame:
                    first_frame = False
                    frame_count = 0
                    start_time = time.time()
                    stream.width = self.width or image.get_width()
                    stream.height = self.height or image.get_height()

                frame = av.VideoFrame.from_ndarray(image.get_data()[:, :, :3], format='bgr24')
                packet = stream.encode(frame)
                container.mux(packet)
                frame_count += 1

                if frame_count % 30 == 0:
                    print(f"Current FPS: {frame_count / (time.time() - start_time):.2f}")
        finally:
            packet = stream.encode(None)
            container.mux(packet)
            container.close()
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Total frames written: {frame_count}")
            print(f"Total time taken: {elapsed_time:.2f} seconds")
            print(f"Average FPS: {frame_count / elapsed_time:.2f}")

    async def run(self):
        thread = threading.Thread(target=self.encode_video)
        thread.start()
        try:
            while True:
                _, record = await self.ins.record.read()
                try:
                    self.queue.put_nowait(record)
                except queue.Full:
                    continue
        finally:
            self.stop_event.set()
            thread.join()


# Test SLCamera system
async def _main():

    camera = SLCamera(fps=15, view=sl.VIEW.SIDE_BY_SIDE, resolution=sl.RESOLUTION.VGA)
    video_dumper = VideoDumper("video.mp4", 15, codec='libx264')
    video_dumper.ins.record = camera.outs.record

    await asyncio.gather(video_dumper.run(), camera.run())

if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("Program interrupted by user, exiting...")
