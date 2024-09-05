# System that starts a webserver that collects tracking data from the WebXR page
# and outputs it further.

import queue
import multiprocessing
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import FileResponse
import uvicorn
import numpy as np

from control import ControlSystem, World
from control.utils import FPSCounter
from geom import Transform3D

def run_server(app, port, ssl_keyfile, ssl_certfile):
    config = uvicorn.Config(app, host="0.0.0.0", port=port,
                            ssl_keyfile=ssl_keyfile, ssl_certfile=ssl_certfile,
                            log_config={'version': 1, 'disable_existing_loggers': False,
                            'handlers': {
                                'file': {
                                    'class': 'logging.FileHandler',
                                    'formatter': 'default',
                                    'filename': 'webxr.log',
                                    'mode': 'w',
                                },
                                'console': {
                                    'class': 'logging.StreamHandler',
                                    'formatter': 'default',
                                }},
                            'loggers': {
                                '': {
                                    'handlers': ['file'],
                                    'level': 'INFO',
                                }
                            },
                            'formatters': {
                                'default': {
                                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                }
                            }},
                )
    server = uvicorn.Server(config)
    server.run()

class WebXR(ControlSystem):
    def __init__(self, world: World, port: int, ssl_keyfile: str = "key.pem", ssl_certfile: str = "cert.pem"):
        super().__init__(world, outputs=["transform", "buttons"])
        self.port = port
        self.ssl_keyfile = ssl_keyfile
        self.ssl_certfile = ssl_certfile
        self.app = FastAPI()
        self.setup_routes()

        self.last_ts = None

        self.data_queue = multiprocessing.Queue()
        self.server_process = None

    def setup_routes(self):
        @self.app.get("/")
        async def root():
            return FileResponse("quest_tracking/static/index.html")

        @self.app.get("/webxr-button.js")
        async def webxr_button():
            return FileResponse("quest_tracking/static/webxr-button.js")

        @self.app.post('/track')
        async def track(request: Request):
            data = await request.json()
            if self.last_ts is None or data['timestamp'] > self.last_ts:
                self.last_ts = data['timestamp']
                self.data_queue.put(data)
            return JSONResponse(content={"success": True})

    def run(self):
        self.server_process = multiprocessing.Process(
            target=run_server,
            args=(self.app, self.port, self.ssl_keyfile, self.ssl_certfile)
        )
        self.server_process.start()

        try:
            fps = FPSCounter("WebXR ")
            while not self.should_stop:
                try:
                    data = self.data_queue.get(timeout=1)  # Wait for 10ms
                    pos = np.array(data['position'])
                    quat = np.array(data['orientation'])
                    self.outs.buttons.write(data['buttons'])
                    self.outs.transform.write(Transform3D(pos, quat))
                    fps.tick()
                except queue.Empty:
                    pass
        finally:
            print("Cancelling WebXR")
            self.server_process.terminate()
            self.server_process.join(timeout=5)
            if self.server_process.is_alive():
                print("WebXR did not terminate in time, terminating forcefully")
                self.server_process.kill()
            print("WebXR cancelled")
