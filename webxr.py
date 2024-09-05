# System that starts a webserver that collects tracking data from the WebXR page
# and outputs it further.

import queue
import multiprocessing
import time
from fastapi import FastAPI, WebSocket
from starlette.responses import FileResponse
import uvicorn
import numpy as np
import json

from control import ControlSystem, World
from control.utils import FPSCounter
from geom import Transform3D


def run_server(data_queue, port, ssl_keyfile, ssl_certfile):
    app = FastAPI()

    @app.get("/")
    async def root():
        return FileResponse("quest_tracking/static/index.html")

    @app.get("/webxr-button.js")
    async def webxr_button():
        return FileResponse("quest_tracking/static/webxr-button.js")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        print("WebSocket connection accepted")
        try:
            fps = FPSCounter("Websocket")
            while True:
                data = await websocket.receive_json()
                data_queue.put(data)
                fps.tick()
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            print("WebSocket connection closed")

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
                                    'class': 'logging.FileHandler',
                                    'formatter': 'default',
                                    'filename': 'webxr.log',
                                    'mode': 'w',
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

        self.data_queue = multiprocessing.Queue()
        self.server_process = None

    def run(self):
        self.server_process = multiprocessing.Process(
            target=run_server,
            args=(self.data_queue, self.port, self.ssl_keyfile, self.ssl_certfile)
        )
        self.server_process.start()

        try:
            fps = FPSCounter("WebXR ")
            while not self.should_stop:
                try:
                    data = self.data_queue.get(timeout=1)
                    pos = np.array(data['position'])
                    quat = np.array(data['orientation'])
                    self.outs.buttons.write(data['buttons'])
                    self.outs.transform.write(Transform3D(pos, quat))
                    fps.tick()
                except queue.Empty:
                    continue
        finally:
            print("Cancelling WebXR")
            self.server_process.terminate()
            self.server_process.join(timeout=5)
            if self.server_process.is_alive():
                print("WebXR did not terminate in time, terminating forcefully")
                self.server_process.kill()
            print("WebXR cancelled")
