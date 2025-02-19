"""Control systems for V4L2 cameras using linuxpy library"""

import asyncio
from typing import AsyncIterator, Optional

import cv2
import numpy as np
from linuxpy.video.device import Device, Frame, PixelFormat

import ironic as ir


@ir.ironic_system(output_ports=['frame'])
class LinuxPyCamera(ir.ControlSystem):
    device_path: str
    width: int
    height: int
    fps: int
    pixel_format: str

    _device: Optional[Device] = None
    _aiter: Optional[AsyncIterator[Frame]] = None
    _fps_counter: ir.utils.FPSCounter
    _frame_queue: asyncio.Queue
    _reader_task: Optional[asyncio.Task] = None
    _stopped: asyncio.Event

    def __init__(self, device_path: str,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30,
                 pixel_format: str = "MJPG"):
        """Video4Linux camera that uses linuxpy library

        Args:
            device_path: Path to video device (e.g. "/dev/video0")
            width: Frame width
            height: Frame height
            fps: Target frames per second
            pixel_format: Pixel format (e.g. "MJPG", "YUYV", "UYVY")
        """
        super().__init__()
        self.device_path = device_path
        self.width = width
        self.height = height
        self.fps = fps
        self.pixel_format = pixel_format

        self._fps_counter = ir.utils.FPSCounter(f"Camera {self.device_path}")
        self._frame_queue = asyncio.Queue(maxsize=1)
        self._reader_task = None
        self._stopped = asyncio.Event()

    async def _frame_reader(self):
        """Background coroutine that reads frames and puts them in the queue"""
        try:
            while not self._stopped.is_set():
                try:
                    frame = await anext(self._aiter)
                    if self._frame_queue.full():
                        try:
                            self._frame_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    await self._frame_queue.put(frame)
                except StopAsyncIteration:
                    break
        except Exception:
            raise
        finally:
            self._stopped.set()

    async def setup(self):
        """Set up the camera device and configure format"""
        self._device = Device(self.device_path)
        self._device.open()

        self._device.set_format(self._device.info.buffers[0],
                                self.width, self.height, self.pixel_format)

        self._device.set_fps(self._device.info.buffers[0], self.fps)
        self._aiter = aiter(self._device)

        self._stopped.clear()
        self._reader_task = asyncio.create_task(self._frame_reader())

    async def cleanup(self):
        """Clean up camera resources"""
        self._stopped.set()
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

        if self._device:
            if self._aiter:
                await self._aiter.aclose()
                self._aiter = None
            self._device.close()
            self._device = None

    async def step(self):
        if self._stopped.is_set():
            return ir.State.FINISHED

        try:
            frame = self._frame_queue.get_nowait()
            self._fps_counter.tick()
            await self.outs.frame.write(ir.Message(data=self._process_frame(frame)))
        except asyncio.QueueEmpty:
            pass
        return ir.State.ALIVE

    def _process_frame(self, frame: Frame) -> dict:
        """Process raw frame into output format"""
        data = np.frombuffer(frame.data, dtype=np.uint8)

        match frame.pixel_format:
            case PixelFormat.MJPEG:
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                return {'image': img}
            case PixelFormat.YUYV:
                data = data.reshape((frame.height, frame.width, 2))
                data = cv2.cvtColor(data, cv2.COLOR_YUV2RGB_YUYV)
            case PixelFormat.UYVY:
                data = data.reshape((frame.height, frame.width, 2))
                data = cv2.cvtColor(data, cv2.COLOR_YUV2RGB_UYVY)
            case _:
                # Assume 3 bytes per pixel (RGB/BGR)
                data = data.reshape((frame.height, frame.width, 3))

        return {'image': data}


if __name__ == "__main__":
    import asyncio
    from tools.video import VideoDumper

    async def _main():
        camera = LinuxPyCamera('/dev/video0')
        system = ir.compose(
            camera,
            VideoDumper("video.mp4", 30, codec='libx264').bind(
                image=ir.utils.map_port(lambda x: x['image'], camera.outs.frame)
            )
        )

        await ir.utils.run_gracefully(system)

    asyncio.run(_main())
