from contextlib import asynccontextmanager
import queue
import logging
import time
import sys
import threading

from fastapi import BackgroundTasks, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import FileResponse
import uvicorn
import franky
from franky import Affine, CartesianMotion, ReferenceType, RealtimeConfig
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as R

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('franka.log'), #, mode='w'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

rr.init("franka", spawn=False)
rr.connect('192.168.10.3:9876')
rr.save("franka.rrd")

def q_inv(q):
    r = R.from_quat([q[1], q[2], q[3], q[0]])
    res = r.inv().as_quat()
    return np.array([res[3], res[0], res[1], res[2]])

def q_mul(q1, q2):
    r1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])
    r2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]])
    res = (r1 * r2).as_quat()
    return np.array([res[3], res[0], res[1], res[2]])

class FrankaRobot:
    def __init__(self, ip: str, relative_dynamics_factor: float = 0.02, gripper_speed: float = 0.02):
        self.robot = franky.Robot(ip) #, realtime_config=RealtimeConfig.Ignore)
        self.robot.relative_dynamics_factor = relative_dynamics_factor
        self.robot.set_collision_behavior(
            [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        )
        self.robot.recover_from_errors()
        motion = franky.JointWaypointMotion([
            franky.JointWaypoint([ 0.0,  -0.3, 0.0, -1.8, 0.0, 1.5,  0.65])])
        self.robot.move(motion)

        self.gripper = franky.Gripper(ip)
        self.gripper_speed = gripper_speed
        self.gripper.homing()
        self.gripper_grasped = False

    @property
    def forward_kinematics(self) -> tuple[np.ndarray, np.ndarray, bool]:
        pos = self.robot.current_pose.end_effector_pose
        return pos.translation, pos.quaternion, self.gripper_grasped

    def inverse_kinematics(self,
                           target_position: np.ndarray,
                           target_orientation: np.ndarray,
                           gripper_grasped: bool) -> bool:
        if self.gripper_grasped != gripper_grasped:
            self.gripper_grasped = gripper_grasped
            if self.gripper_grasped:
                logger.info(f"Grasping at {target_position} {target_orientation}")
                try:
                    self.gripper.grasp(0.0, self.gripper_speed, 7.0, epsilon_outer=1.0)
                except franky.CommandException as e:
                    logger.warning(f"Grasping failed: {e}")
                    return False
            else:
                logger.info(f"Opening at {target_position} {target_orientation}")
                self.gripper.open_async(self.gripper_speed)

        try:
            pos = Affine(translation=target_position, quaternion=target_orientation)
            self.robot.move(CartesianMotion(pos, ReferenceType.Absolute), asynchronous=True)
        except franky.ControlException as e:
            self.robot.recover_from_errors()
            logger.warning(f"IK failed for {target_position} {target_orientation}: {e}")
            return False
        return True

    @property
    def joint_positions(self) -> np.ndarray:
        return self.robot.current_joint_state.position

    def stop(self):
        self.robot.stop()
        self.gripper.open(self.gripper_speed)

class State:
    def __init__(self):
        self.is_tracking = False
        self.tr_diff = None
        self.target = None
        self.robot = None
        self.start_time = None
        self.last_ts = None
        self.latest_request_queue = queue.Queue(maxsize=1)
        self.robot_update_lock = threading.Lock()

state = State()
def get_state():
    return state

@asynccontextmanager
async def lifespan(app: FastAPI):
    state.robot = FrankaRobot("172.168.0.2", 0.2, 0.4)
    state.start_time = time.time()
    yield
    state.robot.stop()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return FileResponse("quest_tracking/static/index.html")

@app.get("/webxr-button.js")
async def webxr_button():
    return FileResponse("quest_tracking/static/webxr-button.js")


def robot_update(state: State, pos: np.ndarray, quat: np.ndarray, but: tuple[bool, bool]):
    track_but, untrack_but, grasp_but, open_but = but

    t, q, grasped = state.robot.forward_kinematics
    for i in range(3):
        rr.log(f"franka/position/{i}", rr.Scalar(t[i]))
    for i in range(4):
        rr.log(f"franka/orientation/{i}", rr.Scalar(q[i]))
    rr.log(f"franka/gripper", rr.Scalar(grasped))

    if track_but:
        state.tr_diff = t - pos, q_mul(q_inv(quat), q)
        if not state.is_tracking:
            logger.info(f'Start tracking at {t} {q}')
        state.is_tracking = True
    elif untrack_but:
        if state.is_tracking:
            logger.info('Stop tracking')

        state.robot.stop()
        state.target = None
        state.is_tracking = False
    elif state.is_tracking:
        if grasped:
            grasped = not open_but
        else:
            grasped = grasp_but

        state.target = pos + state.tr_diff[0], q_mul(quat, state.tr_diff[1]), grasped
        for i in range(3):
            rr.log(f"target/position/{i}", rr.Scalar(state.target[0][i]))
        for i in range(4):
            rr.log(f"target/orientation/{i}", rr.Scalar(state.target[1][i]))

        if state.target is not None and state.is_tracking:
            state.robot.inverse_kinematics(*state.target)

    for i, v in enumerate(state.robot.joint_positions):
        rr.log(f'joints/{i}', rr.Scalar(v))

def process_latest_request(state: State):
    try:
        data = state.latest_request_queue.get(timeout=1)
    except queue.Empty:
        return

    if state.last_ts is None or data['timestamp'] > state.last_ts:
        state.last_ts = data['timestamp']
        with state.robot_update_lock:
            # logger.info(f'buttons: {data["buttons"]}')
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
            # logger.info(f'rotation: {q_mul(res_quat, q_inv(quat))}')
            robot_update(state, pos, res_quat, but)

        state.latest_request_queue.task_done()

@app.post('/track')
async def track(request: Request, background_tasks: BackgroundTasks, state: State = Depends(get_state)):
    data = await request.json()
    try:
        # Try to put the new request data into the queue
        state.latest_request_queue.put_nowait(data)
    except queue.Full:
        # If the queue is full, replace the existing item
        state.latest_request_queue.get_nowait()
        state.latest_request_queue.put_nowait(data)

    background_tasks.add_task(process_latest_request, state)
    return JSONResponse(content={"success": True})

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=5005, ssl_keyfile="key.pem", ssl_certfile="cert.pem")
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
