import asyncio
from typing import Dict
import numpy as np
import pytest

import ironic as ir
from hardware.camera.merge import merge_cameras


@ir.ironic_system(output_ports=['frame'])
class MockCamera(ir.ControlSystem):
    """Mock camera that emits frames on demand"""
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self._frame_queue = asyncio.Queue()

    async def emit_frame(self, frame: Dict[str, np.ndarray], timestamp: int = None):
        """Send a new frame through the output port"""
        await self.outs.frame.write(ir.Message(frame, timestamp))

    async def step(self):
        return ir.State.ALIVE


@pytest.mark.asyncio
async def test_merge_cameras_with_pulse_camera():
    """Test merging cameras when using one of the cameras as pulse source"""
    # Create mock cameras
    cam1 = MockCamera("cam1")
    cam2 = MockCamera("cam2")
    cameras = {"cam1": cam1, "cam2": cam2}

    # Create merged camera system using cam1 as pulse
    merged = merge_cameras(cameras, "cam1")
    await merged.setup()

    # Create frame receiver
    received_frames = []
    async def frame_handler(message):
        received_frames.append(message.data)
    merged.outs.frame.subscribe(frame_handler)

    # Emit test frames
    frame1 = {"image": np.zeros((10, 10, 3))}
    frame2 = {"image": np.ones((10, 10, 3))}

    # First emit frame from cam2 (non-pulse camera)
    await cam2.emit_frame(frame2)
    await asyncio.sleep(0)  # Let event loop process messages
    assert len(received_frames) == 0  # Should not emit until pulse camera sends frame

    # Now emit frame from pulse camera
    await cam1.emit_frame(frame1)
    await asyncio.sleep(0)

    assert len(received_frames) == 1
    merged_frame = received_frames[0]
    assert "cam1.image" in merged_frame
    assert "cam2.image" in merged_frame
    np.testing.assert_array_equal(merged_frame["cam1.image"], frame1["image"])
    np.testing.assert_array_equal(merged_frame["cam2.image"], frame2["image"])


@pytest.mark.asyncio
async def test_merge_cameras_with_external_pulse():
    """Test merging cameras when using external signal as pulse source"""
    # Create mock cameras
    cam1 = MockCamera("cam1")
    cam2 = MockCamera("cam2")
    cameras = {"cam1": cam1, "cam2": cam2}

    # Create pulse port
    pulse_port = ir.OutputPort("pulse")

    # Create merged camera system using external pulse
    merged = merge_cameras(cameras, pulse_port)
    await merged.setup()

    # Create frame receiver
    received_frames = []
    async def frame_handler(message):
        received_frames.append(message.data)
    merged.outs.frame.subscribe(frame_handler)

    # Emit test frames from cameras
    frame1 = {"image": np.zeros((10, 10, 3))}
    frame2 = {"image": np.ones((10, 10, 3))}

    await cam1.emit_frame(frame1)
    await cam2.emit_frame(frame2)
    await asyncio.sleep(0)
    assert len(received_frames) == 0  # Should not emit until pulse signal

    # Send pulse signal
    await pulse_port.write(ir.Message(True))
    await asyncio.sleep(0)

    assert len(received_frames) == 1
    merged_frame = received_frames[0]
    assert "cam1.image" in merged_frame
    assert "cam2.image" in merged_frame
    np.testing.assert_array_equal(merged_frame["cam1.image"], frame1["image"])
    np.testing.assert_array_equal(merged_frame["cam2.image"], frame2["image"])


@pytest.mark.asyncio
async def test_merge_cameras_missing_frame():
    """Test merging cameras when one camera hasn't sent a frame yet"""
    cam1 = MockCamera("cam1")
    cam2 = MockCamera("cam2")
    cameras = {"cam1": cam1, "cam2": cam2}

    merged = merge_cameras(cameras, "cam1")
    await merged.setup()

    received_frames = []
    async def frame_handler(message):
        received_frames.append(message.data)
    merged.outs.frame.subscribe(frame_handler)

    # Only emit frame from pulse camera
    frame1 = {"image": np.zeros((10, 10, 3))}
    await cam1.emit_frame(frame1)
    await asyncio.sleep(0)

    # Should not emit anything when a camera frame is missing
    assert len(received_frames) == 0
