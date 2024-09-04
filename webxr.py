# System that starts a webserver that collects tracking data from the WebXR page
# and outputs it further.

import asyncio
import threading
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import FileResponse
import uvicorn
import numpy as np
from queue import Queue

from control import ControlSystem
from control.utils import FPSCounter
from geom import Transform3D

class WebXR(ControlSystem):
    def __init__(self, port: int, ssl_keyfile: str = "key.pem", ssl_certfile: str = "cert.pem"):
        super().__init__(outputs=["transform", "buttons"])
        self.port = port
        self.ssl_keyfile = ssl_keyfile
        self.ssl_certfile = ssl_certfile
        self.app = FastAPI()
        self.setup_routes()

        self.last_ts = None

        self.data_queue = Queue()
        self.server_thread = None

        config = uvicorn.Config(self.app, host="0.0.0.0", port=self.port,
                                ssl_keyfile=self.ssl_keyfile, ssl_certfile=self.ssl_certfile,
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
        self.server = uvicorn.Server(config)

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

    async def run(self):
        self.server_thread = threading.Thread(target=self.server.run)
        self.server_thread.start()

        try:
            fps = FPSCounter("WebXR")
            while True:
                data = await asyncio.to_thread(self.data_queue.get)
                pos = np.array(data['position'])
                quat = np.array(data['orientation'])

                await self.outs.buttons.write(data['buttons'])
                await self.outs.transform.write(Transform3D(pos, quat))
                fps.tick()
        finally:
            print("Cancelling WebXR")
            self.server.should_exit = True
            self.server.force_exit = True
            self.server_thread.join()
            print("WebXR cancelled")
