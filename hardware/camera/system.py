import ironic as ir
from hardware.camera import Camera


@ir.ironic_system(output_ports=['frame'])
class CameraCS(ir.ControlSystem):
    def __init__(self, camera: Camera):
        super().__init__()
        self.camera = camera

    async def setup(self):
        self.camera.setup()

    async def cleanup(self):
        self.camera.cleanup()

    async def step(self):
        frame, timestamp = self.camera.get_frame()
        await self.outs.frame.write(ir.Message(data=frame, timestamp=timestamp))
        return ir.State.ALIVE
