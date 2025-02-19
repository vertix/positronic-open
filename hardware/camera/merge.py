import asyncio
from typing import Any, Dict, List, Tuple, Union
import ironic as ir

def _merge_frames(cam2frame_dicts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple camera frames into a single dictionary.

    Args:
        cam2frame_dicts: Dictionary mapping camera names to their frame dictionaries.
                        Each frame dictionary maps image names to numpy arrays.

    Returns:
        A merged dictionary with keys of format "{camera_name}.{image_name}".
        Returns ir.NoValue if any frame is ir.NoValue.
    """
    result = {}
    for cam_key, frame_dict in cam2frame_dicts.items():
        if frame_dict == ir.NoValue:
            return ir.NoValue
        result.update({f'{cam_key}.{im_key}': image for im_key, image in frame_dict.items()})
    return result


def merge_on_pulse(cameras: Dict[str, ir.ControlSystem], pulse: ir.OutputPort) -> ir.ControlSystem:
    """Merge multiple cameras into a single virtual camera system, triggered by an external pulse.

    Creates a new control system that combines frames from multiple cameras into a single
    output frame. The merged frames are emitted when triggered by the pulse signal.

    The merged frame contains all images from the input frames, with keys prefixed by the camera name.
    For example, if camera "cam1" outputs {"image": img1} and camera "cam2" outputs {"depth": img2},
    the merged frame will be {"cam1.image": img1, "cam2.depth": img2}.

    If any camera hasn't produced a frame yet when the pulse arrives, the system will output ir.NoValue.

    Args:
        cameras: Dictionary mapping camera names to their control systems. Each camera system must have
                an output port named 'frame' that emits dictionaries mapping image names to numpy arrays.
        pulse: OutputPort that triggers when frames should be merged. When this triggers, the system will
               emit a merged frame containing the most recent frame from each camera.

    Returns:
        A new control system with a single output port 'frame' that emits merged frames.

    Example:
        ```python
        # Merge two cameras using external trigger
        cam1 = SLCamera()  # Outputs {"left": img1, "right": img2}
        cam2 = RealsenseCamera()  # Outputs {"image": img3, "depth": img4}
        cameras = {"cam1": cam1, "cam2": cam2}
        trigger = some_trigger_port

        merged = merge_on_pulse(cameras, trigger)
        # When triggered, will output:
        # {
        #   "cam1.left": img1,
        #   "cam1.right": img2,
        #   "cam2.image": img3,
        #   "cam2.depth": img4
        # }
        ```
    """
    last_values = {name: ir.utils.last_value(camera.outs.frame) for name, camera in cameras.items()}
    merge_dict = ir.utils.properties_dict(**last_values)

    async def merge_frames(_ignored_pulse_data: Any) -> Dict[str, Any]:
        cam2frame_dicts = await merge_dict()
        return _merge_frames(cam2frame_dicts.data)

    return ir.compose(*cameras.values(), outputs={'frame': ir.utils.map_port(merge_frames, pulse)})


def merge_on_camera(main_camera: Tuple[str, ir.ControlSystem], extension_cameras: Dict[str, ir.ControlSystem]) -> ir.ControlSystem:
    """Merge multiple cameras into a single virtual camera system, triggered by a main camera.

    Creates a new control system that combines frames from multiple cameras into a single
    output frame. The merged frames are emitted whenever the main camera produces a new frame.

    The merged frame contains all images from the input frames, with keys prefixed by the camera name.
    For example, if main camera "cam1" outputs {"image": img1} and extension camera "cam2" outputs
    {"depth": img2}, the merged frame will be {"cam1.image": img1, "cam2.depth": img2}.

    If any extension camera hasn't produced a frame yet when the main camera emits a frame,
    the system will output ir.NoValue.

    Args:
        main_camera: Tuple of (name, camera) where camera is the control system that triggers frame merging.
                    The camera must have an output port named 'frame' that emits dictionaries mapping
                    image names to numpy arrays.
        extension_cameras: Dictionary mapping camera names to their control systems. Each camera system
                         must have an output port named 'frame' that emits dictionaries mapping image
                         names to numpy arrays.

    Returns:
        A new control system with a single output port 'frame' that emits merged frames.

    Example:
        ```python
        # Merge cameras using main camera as trigger
        main = SLCamera()  # Outputs {"left": img1, "right": img2}
        ext = RealsenseCamera()  # Outputs {"image": img3, "depth": img4}
        extension_cameras = {"ext": ext}

        merged = merge_on_camera(("main", main), extension_cameras)
        # When main camera emits a frame, will output:
        # {
        #   "main.left": img1,
        #   "main.right": img2,
        #   "ext.image": img3,
        #   "ext.depth": img4
        # }
        ```

    Raises:
        AssertionError: If the main camera name conflicts with an extension camera name.
    """
    last_values = {name: ir.utils.last_value(camera.outs.frame) for name, camera in extension_cameras.items()}
    merge_dict = ir.utils.properties_dict(**last_values)

    name, camera = main_camera
    assert name not in extension_cameras, f"Main camera {name} is also an extension camera"

    async def merge_frames(main_camera_frame: Any) -> Dict[str, Any]:
        cam2frame_dicts = (await merge_dict()).data
        cam2frame_dicts[name] = main_camera_frame
        return _merge_frames(cam2frame_dicts)

    return ir.compose(*extension_cameras.values(), camera, outputs={'frame': ir.utils.map_port(merge_frames, camera.outs.frame)})
