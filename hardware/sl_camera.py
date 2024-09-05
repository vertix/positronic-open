# Control systems for StereoLabs cameras

from dataclasses import dataclass
from typing import Optional

import pyzed.sl as sl

from control import ControlSystem, utils, World, MainThreadWorld
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

    def run(self):
        zed = sl.Camera()
        zed.open(self.init_params)
        SUCCESS = sl.ERROR_CODE.SUCCESS
        TIME_REF_IMAGE = sl.TIME_REFERENCE.IMAGE

        try:
            fps = utils.FPSCounter("Camera")
            while not self.should_stop:
                result = zed.grab()
                fps.tick()
                if result != SUCCESS:
                    # TODO: Should we be more specific about the error?
                    # See(https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1ERROR__CODE.html)
                    self.outs.record.write(Record(success=False))
                    continue

                image = sl.Mat()
                if zed.retrieve_image(image, self.view) != SUCCESS:
                    self.outs.record.write(Record(success=False))
                    continue

                ts_ms = zed.get_timestamp(TIME_REF_IMAGE).get_milliseconds()
                self.outs.record.write(Record(success=True, image=image), timestamp=ts_ms)
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
        return record.image.get_data()[:, :, :3]

    video_dumper.ins.image = extract_np_image(camera.outs.record)

    world.run()


if __name__ == "__main__":
    main()
