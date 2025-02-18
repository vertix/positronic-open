import asyncio
from typing import Any, Dict, List, Union
import ironic as ir


def merge_cameras(cameras: Dict[str, ir.ControlSystem], pulse: Union[str, ir.OutputPort]) -> ir.ControlSystem:
    """Merge multiple cameras into a single virtual camera system.

    This function creates a new control system that combines frames from multiple cameras into a single
    output frame. The merged frames are emitted when triggered by a pulse signal, which can be either
    one of the cameras or an external signal.

    The merged frame contains all images from the input frames, with keys prefixed by the camera name.
    For example, if camera "cam1" outputs {"image": img1} and camera "cam2" outputs {"depth": img2},
    the merged frame will be {"cam1.image": img1, "cam2.depth": img2}.

    If any camera hasn't produced a frame yet when the pulse arrives, the system will output ir.NoValue.

    Args:
        cameras: Dictionary mapping camera names to their control systems. Each camera system must have
                an output port named 'frame' that emits dictionaries mapping image names to numpy arrays.
        pulse: Either a camera name from the cameras dict (to use that camera's frames as trigger) or
               an OutputPort (to use external trigger signal). When this triggers, the system will emit
               a merged frame containing the most recent frame from each camera.

    Returns:
        A new control system with a single output port 'frame' that emits merged frames.

    Example:
        ```python
        # Merge two cameras using cam1 as pulse source
        cam1 = SLCamera()  # Outputs {"left": img1, "right": img2}
        cam2 = RealsenseCamera()  # Outputs {"image": img3, "depth": img4}
        cameras = {"cam1": cam1, "cam2": cam2}

        merged = merge_cameras(cameras, "cam1")
        # Will output:
        # {
        #   "cam1.left": img1,
        #   "cam1.right": img2,
        #   "cam2.image": img3,
        #   "cam2.depth": img4
        # }
        ```
    """
    def merge_frames(cam2frame_dicts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        result = {}
        for cam_key, frame_dict in cam2frame_dicts.items():
            if frame_dict == ir.NoValue:
                return ir.NoValue
            result.update({f'{cam_key}.{im_key}': image for im_key, image in frame_dict.items()})
        return result

    last_values = {name: ir.utils.last_value(camera.outs.frame) for name, camera in cameras.items()}
    if isinstance(pulse, str):
        del last_values[pulse]
        pulse_camera = cameras[pulse]
        out_port = ir.OutputPort("merged_camera_frame", pulse_camera.outs.frame.parent_system)

        async def handler(message: ir.Message):
            frame_msgs = await asyncio.gather(*[prop() for prop in last_values.values()])
            cam2frame_dicts = {cam_name: frame.data for cam_name, frame in zip(last_values.keys(), frame_msgs)}
            cam2frame_dicts[pulse] = message.data
            result = merge_frames(cam2frame_dicts)
            if result != ir.NoValue:
                await out_port.write(ir.Message(result, message.timestamp))

        pulse_camera.outs.frame.subscribe(handler)
    else:
        out_port = ir.OutputPort("merged_camera_frame", pulse.parent_system)
        async def handler(message: ir.Message):
            frame_msgs = await asyncio.gather(*[prop() for prop in last_values.values()])
            cam2frame_dicts = {cam_name: frame.data for cam_name, frame in zip(last_values.keys(), frame_msgs)}
            result = merge_frames(cam2frame_dicts)
            if result != ir.NoValue:
                await out_port.write(ir.Message(result, message.timestamp))

        pulse.subscribe(handler)

    return ir.compose(*cameras.values(), outputs={'frame': out_port})
