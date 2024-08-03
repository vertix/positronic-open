import queue
import logging
import threading
import time
import sys

from flask import Flask, request, jsonify
import franky
from franky import Affine, CartesianMotion, ReferenceType, RealtimeConfig
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as R


rr.init("franka", spawn=False)
rr.connect('192.168.10.3:9876')
rr.save("franka.rrd")

app = Flask(__name__, static_url_path='', static_folder='quest_tracking/static')
tracker_position = {"pos": None, "quat": None, "but": (False, False)}
tracker_position_lock = threading.Lock()

def q_inv(q):
    r = R.from_quat([q[1], q[2], q[3], q[0]])
    res = r.inv().as_quat()
    return np.array([res[3], res[0], res[1], res[2]])

def q_mul(q1, q2):
    r1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])
    r2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]])
    res = (r1 * r2).as_quat()
    return np.array([res[3], res[0], res[1], res[2]])

latest_request_queue = queue.Queue(maxsize=1)

@app.route('/track', methods=['POST'])
def track():
    data = request.json
    try:
        # Try to put the new request data into the queue
        latest_request_queue.put_nowait(data)
    except queue.Full:
        # If the queue is full, replace the existing item
        latest_request_queue.get_nowait()
        latest_request_queue.put_nowait(data)
    return jsonify(success=True)

def process_latest_request():
    last_ts = None
    while True:
        try:
            data = latest_request_queue.get(timeout=1)
        except queue.Empty:
            continue

        if last_ts is None or data['timestamp'] > last_ts:
            last_ts = data['timestamp']
            pos = np.array(data['position'])
            quat = np.array(data['orientation'])
            but = (data['buttons'][4], data['buttons'][5]) if len(data['buttons']) > 5 else (False, False)

            pos = np.array([-pos[2], -pos[0], pos[1]])
            quat = np.array([quat[0], -quat[3], -quat[1], quat[2]])
            # Rotate quat 90 degrees around Y axis
            rotation_y_90 = np.array([np.cos(np.pi/4), 0, np.sin(np.pi/4), 0])  # 90 degrees rotation around Y
            quat = q_mul(rotation_y_90, quat)

            with tracker_position_lock:
                tracker_position['pos'] = pos
                tracker_position['quat'] = quat
                tracker_position['but'] = but

        latest_request_queue.task_done()

# The abstract class for a Robot, that may have different implementations
# for different simulators and different physical robots.
class Robot:
    def __init__(self):
        pass

    @property
    def forward_kinematics(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the end effector position and orientation.

        Returns:
            tuple: A tuple containing two numpy arrays:
                - position (np.ndarray): 3D position vector
                - orientation (np.ndarray): 4D quaternion
        """
        pass

    def inverse_kinematics(self, target_position: np.ndarray, target_orientation: np.ndarray) -> bool:
        """
        Runs the inverse kinematics solver for the robot, trying to achieve the given target position and orientation.

        Args:
            target_position (np.ndarray): 3D position vector
            target_orientation (np.ndarray): 4D quaternion

        Returns:
            bool: True if the inverse kinematics solution was found and applied, False otherwise.
        """
        pass

    @property
    def joint_positions(self) -> np.ndarray:
        """
        Returns the current joint positions of the robot.

        Returns:
            np.ndarray: joint position vector
        """
        pass


class FrankaRobot(Robot):
    def __init__(self, ip: str, relative_dynamics_factor: float = 0.02):
        self.robot = franky.Robot(ip, realtime_config=RealtimeConfig.Ignore)
        self.robot.relative_dynamics_factor = relative_dynamics_factor
        self.robot.recover_from_errors()
        motion = franky.JointWaypointMotion([
            franky.JointWaypoint([ 0.0,  -0.3, 0.0, -1.8, 0.0, 1.5,  0.6])])
        self.robot.move(motion)

    @property
    def forward_kinematics(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.robot.current_pose.end_effector_pose
        return pos.translation, pos.quaternion

    def inverse_kinematics(self, target_position: np.ndarray, target_orientation: np.ndarray) -> bool:
        pos = Affine(translation=target_position, quaternion=target_orientation)
        self.robot.move(CartesianMotion(pos, ReferenceType.Absolute), asynchronous=True)
        return True

    @property
    def joint_positions(self) -> np.ndarray:
        return self.robot.current_joint_state.position

    def stop(self):
        motion = franky.CartesianPoseStopMotion()
        self.robot.move(motion)

def main():
    is_tracking = False
    tr_diff = None
    franka_origin = None
    target = None

    robot = FrankaRobot("172.168.0.2", 0.6)

    start = time.time()

    while True:
        rr.set_time_seconds("time", time.time() - start)

        with tracker_position_lock:
            pos, quat, but = tracker_position['pos'], tracker_position['quat'], tracker_position['but']

        if but[0]:
            franka_origin = robot.forward_kinematics
            tr_diff = franka_origin[0] - pos, q_mul(q_inv(quat), franka_origin[1])
            if not is_tracking:
                print(f'Start tracking {tr_diff}')
            is_tracking = True
        elif but[1]:
            if is_tracking:
                print('Stop tracking')

            robot.stop()
            target = None
            is_tracking = False
        elif is_tracking:
            target = pos + tr_diff[0], q_mul(quat, tr_diff[1])  # quat #
            for i in range(3):
                rr.log(f"target/position/{i}", rr.Scalar(target[0][i]))
            for i in range(4):
                rr.log(f"target/orientation/{i}", rr.Scalar(target[1][i]))

        if target is not None and is_tracking:
            # if not robot.inverse_kinematics(target[0], target[1]):
            if not robot.inverse_kinematics(target[0], robot.forward_kinematics[1]):
                print(f"IK failed for {target}")

        t, q = robot.forward_kinematics
        for i in range(3):
            rr.log(f"franka/position/{i}", rr.Scalar(t[i]))
        for i in range(4):
            rr.log(f"franka/orientation/{i}", rr.Scalar(q[i]))

        for i, v in enumerate(robot.joint_positions):
            rr.log(f'joints/{i}', rr.Scalar(v))


if __name__ == "__main__":
    def run_server():
        log = logging.getLogger('werkzeug')
        # log.setLevel(logging.ERROR)  # Set the logging level to ERROR to suppress INFO logs
        log.setLevel(logging.CRITICAL)
        app.logger.setLevel(logging.CRITICAL)

        app.run(
            host='0.0.0.0',
            port=5005,
            ssl_context=('cert.pem', 'key.pem'),
            # debug=True
        )

    flask_thread = threading.Thread(target=run_server)
    flask_thread.start()
    worker_thread = threading.Thread(target=process_latest_request)
    worker_thread.start()
    main()

    # flask_thread.join()
    # worker_thread.join()

print('Finished')
