# System that starts a webserver that collects tracking data from the WebXR page
# and outputs it further.

import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import FileResponse
import uvicorn
import numpy as np
from scipy.spatial.transform import Rotation

from control import ControlSystem
from geom import Transform, q_mul


# 2. Утолщить основу фланца, сделать дырки под болты
# 3. Расширить крепления под болты гриппера
# 4. Чуть расширить отверстия под болты гриппера

class WebXR(ControlSystem):
    def __init__(self, port: int, ssl_keyfile: str = "key.pem", ssl_certfile: str = "cert.pem"):
        super().__init__(outputs=["transform", "buttons"])
        self.port = port
        self.ssl_keyfile = ssl_keyfile
        self.ssl_certfile = ssl_certfile

    async def run(self):
        app = FastAPI()
        last_ts = None

        @app.get("/")
        async def root():
            return FileResponse("quest_tracking/static/index.html")

        @app.get("/webxr-button.js")
        async def webxr_button():
            return FileResponse("quest_tracking/static/webxr-button.js")

        @app.post('/track')
        async def track(request: Request):
            nonlocal last_ts
            data = await request.json()
            if last_ts is None or data['timestamp'] > last_ts:
                last_ts = data['timestamp']
                pos = np.array(data['position'])
                quat = np.array(data['orientation'])
                if len(data['buttons']) > 6:
                    but = (data['buttons'][4], data['buttons'][5], data['buttons'][0], data['buttons'][1])
                else:
                    but = (False, False, False, False)

                pos = np.array([pos[2], pos[0], pos[1]])
                quat = np.array([quat[0], quat[3], quat[1], quat[2]])

                # Don't ask my why these transformations, I just got them
                # Rotate quat 90 degrees around Y axis
                rotation_y_90 = np.array([np.cos(-np.pi/4), 0, np.sin(-np.pi/4), 0])
                res_quat = q_mul(rotation_y_90, quat)
                res_quat = np.array([-res_quat[0], res_quat[1], res_quat[2], res_quat[3]])

                await self.outs.buttons.write(but)
                await self.outs.transform.write(Transform(pos, res_quat))

            return JSONResponse(content={"success": True})

        config = uvicorn.Config(app, host="0.0.0.0", port=self.port,
                                ssl_keyfile=self.ssl_keyfile, ssl_certfile=self.ssl_certfile,
                                log_config={'version': 1, 'disable_existing_loggers': False,
                                'handlers': {
                                    'file': {
                                        'class': 'logging.FileHandler',
                                        'formatter': 'default',
                                        'filename': 'webxr.log',
                                        'mode': 'w',  # Overwrite the log file on every start
                                    },
                                    'console': {
                                        'class': 'logging.StreamHandler',
                                        'formatter': 'default',
                                    }},
                                'loggers': {
                                    '': {
                                        'handlers': ['file'], # 'console'],
                                        'level': 'INFO',
                                    }
                                },
                                'formatters': {
                                    'default': {
                                        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                    }
                                }},
                    )
        try:
            server = uvicorn.Server(config)
            await server.serve()
        except asyncio.CancelledError:
            pass
