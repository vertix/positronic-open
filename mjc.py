from dm_control import mujoco
from dm_control.utils import inverse_kinematics as ik

import numpy as np
import os
import threading
from flask import Flask, request, jsonify
from scipy.spatial.transform import Rotation as R
import rerun as rr

rr.init("franka_sim", spawn=False)
rr.save("mujoco_sim.rr")
# rr.connect()

app = Flask(__name__, static_url_path='', static_folder='quest_tracking/static')
tracker_position = {"transform": None, "but": False}
tracker_position_lock = threading.Lock()

last_ts = None
last_ts_lock = threading.Lock()

class Transform3D:
    def __init__(self, position, orientation):
        self.position = position  # np.array([x, y, z])
        self.orientation = R.from_quat(orientation)  # np.quaternion([w, x, y, z])

    def __mul__(self, other):
        new_position = self.position + self.orientation.apply(other.position)
        new_orientation = self.orientation * other.orientation
        return Transform3D(new_position, new_orientation.as_quat())

    def inverse(self):
        inv_orientation = self.orientation.inv()
        inv_position = -inv_orientation.apply(self.position)
        return Transform3D(inv_position, inv_orientation.as_quat())

    def __str__(self):
        pos_str = ", ".join([f"{v:.3f}" for v in self.position])
        ori_str = ", ".join([f"{v:.3f}" for v in self.orientation.as_quat()])
        return f"Tr3D(position={pos_str}, orientation={ori_str})"

    @property
    def rerun(self):
        q = self.orientation.as_quat()
        return rr.Transform3D(translation=self.position,
                              rotation=rr.Quaternion(xyzw=[q[1], q[2], q[3], q[0]]))

@app.route('/track', methods=['POST'])
def track():
    global last_ts
    data = request.json

    with last_ts_lock:
        if (last_ts is None or data['timestamp'] - last_ts > 100) and \
           len(data['position']) > 0 and len(data['buttons']) > 0:
            with tracker_position_lock:
                tracker_position['transform'] = Transform3D(
                    position=np.array(data['position']),
                    orientation=np.array(data['orientation']))
                tracker_position['but'] = data['buttons'][4] if len(data['buttons']) > 4 else False
            last_ts = data['timestamp']

    return jsonify(success=True)

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

    tracking_shift = None
    tracker_origin = None
    franka_origin = None
    target = None

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            rr.set_time_seconds("time", data.time)

            with tracker_position_lock:
                if tracker_position['but']:
                    tracker_origin = tracker_position['transform']
                    franka_origin = Transform3D(data.qpos[:3], data.qpos[3:7])
                    tracking_shift = Transform3D(data.qpos[:3], data.qpos[3:7]) * tracker_position['transform'].inverse()
                    print(f"tracking_shift: {tracking_shift}")
                elif tracking_shift is not None:
                    target = tracking_shift * tracker_position['transform']

                    tracking_move = tracker_origin.inverse() * tracker_position['transform']
                    franka_move = franka_origin.inverse() * Transform3D(data.qpos[:3], data.qpos[3:7])
                    print(f"TD: {tracking_move} FD: {franka_move}")
                    log_transform("move/tracking", tracking_move)
                    log_transform("move/franka", franka_move)

            site_xpos = data.site_xpos[site_id]
            site_quat = xmat_to_quat(data.site_xmat[site_id])
            log = f"Current: {Transform3D(site_xpos, site_quat)}. "
            rr.log("position/current", Transform3D(site_xpos, site_quat).rerun)
            for i, v in zip('xyz', site_xpos):
                rr.log(f"translation/current/{i}", rr.Scalar(v))
            for i, v in zip('wxyz', site_quat):
                rr.log(f"rotation/current/{i}", rr.Scalar(v))

            if target is not None:
                result = ik.qpos_from_site_pose(
                    physics=physics,
                    site_name='end_effector',
                    target_pos=target.position,
                    target_quat=target.orientation.as_quat(),
                    joint_names=joints,
                )

                rr.log("position/target", target.rerun)
                for i, v in zip('xyz', target.position):
                    rr.log(f"translation/target/{i}", rr.Scalar(v))
                for i, v in zip('wxyz', target.orientation.as_quat()):
                    rr.log(f"rotation/target/{i}", rr.Scalar(v))

                if result.success:
                    data.ctrl[:7] = result.qpos[:7]
                else:
                    log += f"IK failed: target: {target}"
                    print(log)

            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    flask_thread = threading.Thread(
        target=app.run, args=('0.0.0.0', 5005), kwargs={'ssl_context': ('cert.pem', 'key.pem')})
    flask_thread.start()

    main()
    flask_thread.join()

print('Finished')
