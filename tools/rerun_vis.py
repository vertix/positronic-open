import numpy as np
import rerun as rr
from typing import Dict

import ironic as ir


def wxyz_to_xyzw(wxyz: np.ndarray) -> np.ndarray:
    return np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])

@ir.ironic_system(
    input_ports=['frame'],
    input_props=['ext_force_ee', 'ext_force_base', 'robot_position'],
)
class RerunVisualiser(ir.ControlSystem):
    """Control system for visualizing data streams using rerun."""

    def __init__(self) -> None:
        super().__init__()
        self.recording_id = 0

    async def setup(self):
        """Initialize rerun recording."""
        rr.init("Data Collection Visualizer", spawn=True)

    @ir.on_message('frame')
    async def on_frame(self, msg: ir.Message):
        """Handle incoming camera frames."""
        frames: Dict[str, np.ndarray] = msg.data
        rr.set_time_seconds("camera", seconds=msg.timestamp)

        # Handle RGB image
        if 'image' in frames:
            rr.log("camera/rgb", rr.Image(frames['image']).compress())

        if 'depth' in frames:
            depth_m = frames['depth']
            rr.log("camera/depth", rr.Image(depth_m))

        if 'infrared_1' in frames:
            rr.log("camera/infrared_1", rr.Image(frames['infrared_1']).compress())
        if 'infrared_2' in frames:
            rr.log("camera/infrared_2", rr.Image(frames['infrared_2']).compress())

        ee_force = np.array((await self.ins.ext_force_ee()).data)

        rr.log("forces/end_effector/x", rr.Scalar(ee_force[0]))
        rr.log("forces/end_effector/y", rr.Scalar(ee_force[1]))
        rr.log("forces/end_effector/z", rr.Scalar(ee_force[2]))

        base_force = (await self.ins.ext_force_base()).data
        rr.log("forces/base/x", rr.Scalar(base_force[0]))
        rr.log("forces/base/y", rr.Scalar(base_force[1]))
        rr.log("forces/base/z", rr.Scalar(base_force[2]))

        robot_position = (await self.ins.robot_position()).data
        rr_robot_position = rr.Transform3D(
            translation=robot_position.translation.copy(),
            rotation=rr.Quaternion(xyzw=wxyz_to_xyzw(robot_position.quaternion))
        )
        rr.log("robot/position", rr_robot_position)
        rr.log("robot/ee_force", rr.Arrows3D(
            origins=robot_position.translation.copy(),
            vectors=ee_force[:3].copy()
        ))
