import av

from control import ControlSystem, World
from control.utils import FPSCounter


class VideoDumper(ControlSystem):
    def __init__(self, world: World, filename: str, fps: int, width: int = None, height: int = None, codec: str = 'libx264'):
        super().__init__(world, inputs=['image'], outputs=[])
        self.filename = filename
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec

    def run(self):
        container = av.open(self.filename, mode='w', format='mp4')
        stream = container.add_stream(self.codec, rate=self.fps)
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '27', 'g': '2', 'preset': 'ultrafast', 'tune': 'zerolatency'}
        first_frame = True

        try:
            fps = None
            for _, image in self.ins.image.read_until_stop():
                if image is None:
                    continue

                if first_frame:
                    fps = FPSCounter("VideoDumper")
                    first_frame = False
                    stream.width = self.width or image.shape[1]
                    stream.height = self.height or image.shape[0]

                frame = av.VideoFrame.from_ndarray(image, format='bgr24')
                packet = stream.encode(frame)
                container.mux(packet)

                fps.tick()
        finally:
            packet = stream.encode(None)
            container.mux(packet)
            container.close()
