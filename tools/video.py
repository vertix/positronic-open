import queue
import time
import threading

import av

from control import ControlSystem

class VideoDumper(ControlSystem):
    def __init__(self, filename: str, fps: int, width: int = None, height: int = None, codec: str = 'libx264'):
        super().__init__(inputs=['image'], outputs=[])
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
                image = self.queue.get()
                if image is None:
                    continue

                if first_frame:
                    first_frame = False
                    frame_count = 0
                    start_time = time.time()
                    stream.width = self.width or image.shape[1]
                    stream.height = self.height or image.shape[0]

                frame = av.VideoFrame.from_ndarray(image, format='bgr24')
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
                _, image = await self.ins.image.read()
                try:
                    self.queue.put_nowait(image)
                except queue.Full:
                    continue
        finally:
            self.stop_event.set()
            thread.join()
