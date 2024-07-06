import queue
import threading

from dm_control import mujoco
from dm_control.utils import inverse_kinematics as ik
from flask import Flask, request, jsonify
import numpy as np
import rerun as rr

rr.init("mujoco_sim", spawn=False)
# rr.connect()
rr.save("mujoco.rrd")


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
            # Compute updated rotation quaternion given coordinate system transformation above
            # Transform the orientation to the new coordinate system, where Z points up, X points forward, and Y points left
            transformed_quat = np.array([quat[3], quat[2], -quat[0], quat[1]])
            quat = transformed_quat

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

def main():
    model = mujoco.MjModel.from_xml_path("assets/mujoco/scene.xml")
    data = mujoco.MjData(model)
    physics = mujoco.Physics.from_model(data)
    joints = [f'joint{i}' for i in range(1, 8)]
    site_id = model.site('end_effector').id
    mocap_ee_id, mocap_tracker_id = 0, 1

    def get_real_ee_position():
        return data.site_xpos[site_id], xmat_to_quat(data.site_xmat[site_id])
        # return data.mocap_pos[mocap_ee_id], data.mocap_quat[mocap_ee_id]

    is_tracking = False
    tr_diff = None
    tracker_origin = None
    franka_origin = None
    target = None

    with mujoco.viewer.launch_passive(model, data, show_right_ui=False) as viewer:
        rr.set_time_seconds("time", data.time)
        while viewer.is_running():
            with tracker_position_lock:
                pos, quat, but = tracker_position['pos'], tracker_position['quat'], tracker_position['but']
                quat = np.array([1.0, 0., 0., 0.])

            if but[0]:
                franka_origin = get_real_ee_position()
                tr_diff = franka_origin[0] - pos, q_mul(q_inv(quat), franka_origin[1])
                is_tracking = True
            elif but[1]:
                is_tracking = False
            elif is_tracking:
                target = pos + tr_diff[0], q_mul(quat, tr_diff[1])

            data.mocap_pos[mocap_tracker_id] = pos
            data.mocap_quat[mocap_tracker_id] = quat

            if target is not None and is_tracking:
                data.mocap_pos[mocap_ee_id] = target[0]
                data.mocap_quat[mocap_ee_id] = target[1]

                result = ik.qpos_from_site_pose(
                    physics=physics,
                    site_name='end_effector',
                    target_pos=target[0],
                    # target_quat=target[1],
                    joint_names=joints,
                )

                if result.success:
                    for i in range(7):
                        rr.log(f'ik/{i}', rr.Scalar(result.qpos[i]))
                        rr.log(f'ctrl/{i}', rr.Scalar(data.ctrl[i]))

                    data.ctrl[:7] = result.qpos[:7]
                else:
                    print(f"IK failed: target: {target}")

            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)  # Set the logging level to ERROR to suppress INFO logs

    # import cProfile
    # with cProfile.Profile() as pr:
    flask_thread = threading.Thread(
        target=app.run, args=('0.0.0.0', 5005), kwargs={'ssl_context': ('cert.pem', 'key.pem')})
    flask_thread.start()
    worker_thread = threading.Thread(target=process_latest_request)
    worker_thread.start()
    main()
    # pr.dump_stats('profile.pstat')

    flask_thread.join()
    worker_thread.join()

print('Finished')
