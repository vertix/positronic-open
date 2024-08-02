import queue
import threading

from dm_control import mujoco
from dm_control.utils import inverse_kinematics as ik
from flask import Flask, request, jsonify
import numpy as np
import rerun as rr

rr.init("mujoco_sim", spawn=True)
rr.connect()
# rr.save("mujoco.rrd")


app = Flask(__name__, static_url_path='', static_folder='quest_tracking/static')
tracker_position = {"pos": None, "quat": None, "but": (False, False)}
tracker_position_lock = threading.Lock()

def q_inv(q):
    result = np.zeros(4)
    mujoco.mju_negQuat(result, q)
    return result

def q_mul(q1, q2):
    result = np.zeros(4)
    mujoco.mju_mulQuat(result, q1, q2)
    return result

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

def log_transform(name, transform):
    for i, v in zip('xyz', transform.position):
        rr.log(f"{name}/position/{i}", rr.Scalar(v))
    for i, v in zip('wxyz', transform.orientation.as_quat()):
        rr.log(f"{name}/orientation/{i}", rr.Scalar(v))

def xmat_to_quat(xmat):
    site_quat = np.empty(4)
    mujoco.mju_mat2Quat(site_quat, xmat)
    return site_quat


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

    @property
    def inverse_kinematics(self, target_position: np.ndarray, target_orientation: np.ndarray) -> bool:
        """
        Returns the inverse kinematics solution for the robot.

        Returns:
            bool: True if the inverse kinematics solution was found and applied, False otherwise.
        """
        pass


class MujocoFrankaRobot(Robot):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.ee_id = model.site('end_effector').id
        self.physics = mujoco.Physics.from_model(data)
        self.joints = [f'joint{i}' for i in range(1, 8)]

    def forward_kinematics(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.site_xpos[self.ee_id], xmat_to_quat(self.data.site_xmat[self.ee_id])

    def inverse_kinematics(self, target_position: np.ndarray, target_orientation: np.ndarray) -> bool:
        result = ik.qpos_from_site_pose(
            physics=self.physics,
            site_name='end_effector',
            target_pos=target_position,
            target_quat=target_orientation,
            joint_names=self.joints,
            rot_weight=0.5,
        )
        if result.success:
            # TODO: This highly relies on the order of the joints in the model
            self.data.ctrl[:7] = result.qpos[:7]
            for i in range(7):
                rr.log(f'ik/qpos/{i}', rr.Scalar(result.qpos[i]))
            rr.log('ik/err_norm', rr.Scalar(result.err_norm))

            return True
        return False

def main():
    model = mujoco.MjModel.from_xml_path("assets/mujoco/scene.xml")
    data = mujoco.MjData(model)
    mocap_ee_id, mocap_tracker_id = 0, 1

    robot = MujocoFrankaRobot(model, data)

    is_tracking = False
    tr_diff = None
    franka_origin = None
    target = None

    with mujoco.viewer.launch_passive(model, data, show_right_ui=False) as viewer, open('simulation_log.txt', 'a') as log_file:
        while viewer.is_running():
            rr.set_time_seconds("time", data.time)

            with tracker_position_lock:
                pos, quat, but = tracker_position['pos'], tracker_position['quat'], tracker_position['but']

            if but[0]:
                franka_origin = robot.forward_kinematics()
                tr_diff = franka_origin[0] - pos, q_mul(q_inv(quat), franka_origin[1])
                is_tracking = True
            elif but[1]:
                is_tracking = False
            elif is_tracking:
                target = pos + tr_diff[0], quat # q_mul(quat, tr_diff[1])  # quat #
                for i in range(3):
                    rr.log(f"target/position/{i}", rr.Scalar(target[0][i]))
                for i in range(4):
                    rr.log(f"target/orientation/{i}", rr.Scalar(target[1][i]))

            data.mocap_pos[mocap_tracker_id] = pos
            data.mocap_quat[mocap_tracker_id] = quat

            if target is not None and is_tracking:
                data.mocap_pos[mocap_ee_id] = target[0]
                data.mocap_quat[mocap_ee_id] = target[1]

                with viewer.lock():
                    if not robot.inverse_kinematics(target[0], target[1]):
                        print(f"IK failed for {target}")

            for i in range(7):
                rr.log(f'ctrl/{i}', rr.Scalar(data.ctrl[i]))

            mujoco.mj_step(model, data)

            viewer.sync()

if __name__ == "__main__":
    def run_server():
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)  # Set the logging level to ERROR to suppress INFO logs
        app.run(
            host='0.0.0.0',
            port=5005,
            ssl_context=('cert.pem', 'key.pem')
        )

    flask_thread = threading.Thread(target=run_server)
    flask_thread.start()
    worker_thread = threading.Thread(target=process_latest_request)
    worker_thread.start()
    main()

    flask_thread.join()
    worker_thread.join()

print('Finished')
