import ironic as ir
from hardware.camera import Camera


@ir.control_system(output_ports=['frame'])
class CameraCS(ir.ControlSystem):
    def __init__(self, camera: Camera):
        self.camera = camera

    def setup(self):
        self.camera.setup()

    def cleanup(self):
        self.camera.cleanup()

    async def step(self):
        frame, timestamp = self.camera.get_frame()
        await self.outs.frame.write(ir.Message(data=frame, timestamp=timestamp))
        return ir.State.ALIVE
