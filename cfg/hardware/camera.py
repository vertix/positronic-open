from typing import Dict, Union
from drivers.camera.merge import merge_on_pulse

from hydra_zen import builds, store, zen

import ironic as ir

def _linux_camera(device_path: str,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30,
                 pixel_format: str = "MJPG"):
    from drivers.camera.linuxpy_video import LinuxPyCamera
    return LinuxPyCamera(device_path, width, height, fps, pixel_format)

def _luxonis_camera(*args, **kwargs):
    from drivers.camera.luxonis import LuxonisCamera
    return LuxonisCamera(*args, **kwargs)


linux_camera = builds(_linux_camera, populate_full_signature=True)
luxonis_camera = builds(_luxonis_camera, populate_full_signature=True)


merge_camera = builds(merge_on_pulse, populate_full_signature=True)


cam_store = store(group="camera")
cam_store(
    linux_camera(device_path="/dev/video0", width=640, height=480, fps=30),
    name='video0'
)

cam_store(
    linux_camera(device_path="/dev/video2", width=640, height=480, fps=30),
    name='video2'
)

# cam_store(luxonis_camera(fps=60), name='luxonis')
cam_store(
    merge_camera(
        cameras={'video0': store.get_entry('camera', 'video0')['node'],
                 'video2': store.get_entry('camera', 'video2')['node']},
        fps=30,
    ),
    name='merged'
)
