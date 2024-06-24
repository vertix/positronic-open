import os
import threading

from flask import Flask, request, jsonify
import numpy as np
from scipy.spatial.transform import Rotation as R
import rerun as rr
from rerun.components import Vector3D

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni
from omni.isaac.core import World
from omni.isaac.sensor import Camera
from omni.isaac.franka import Franka, KinematicsSolver
from omni.isaac.core.objects import DynamicCuboid

from pxr import Usd, UsdGeom

rr.init("franka_sim", spawn=False)
rr.connect()
# rr.save("run.rrd")

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
        # new_position = self.position + other.position  # Ignore orientation for now
        new_orientation = self.orientation * other.orientation
        return Transform3D(new_position, new_orientation.as_quat())

    def inverse(self):
        inv_orientation = self.orientation.inv()
        inv_position = -inv_orientation.apply(self.position)
        # inv_position = -self.position  # Ignore orientation for now
        return Transform3D(inv_position, inv_orientation.as_quat())

    def __str__(self):
        pos_str = ", ".join([f"{v:.3f}" for v in self.position])
        ori_str = ", ".join([f"{v:.3f}" for v in self.orientation.as_quat()])
        return f"Tr3D(position={pos_str}, orientation={ori_str})"
        # return f"Tr3D(position={pos_str})"

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
                    orientation=np.array(data['orientation'])
                )
                tracker_position['but'] = data['buttons'][4]
            last_ts = data['timestamp']

    return jsonify(success=True)

def log_transform(name, transform):
    for i, v in zip('xyz', transform.position):
        rr.log(f"{name}/position/{i}", rr.Scalar(v))
    for i, v in zip('wxyz', transform.orientation.as_quat()):
        rr.log(f"{name}/orientation/{i}", rr.Scalar(v))


def main():
    current_directory = os.getcwd()
    stage_path = os.path.join(current_directory, 'assets/franka_table.usda')
    omni.usd.get_context().open_stage(stage_path)
    omni.kit.app.get_app().update()

    world = World(stage_units_in_meters=1.0)
    world.initialize_physics()
    world.reset()

    stage = omni.usd.get_context().get_stage()
    franka = Franka(prim_path="/World/Franka", name="Franka")
    franka.initialize()

    controller = KinematicsSolver(franka)
    # We don't set the robot base pose here because we want tracking to work in the robot's local frame
    # controller.get_kinematics_solver().set_robot_base_pose(*franka.get_world_pose())
    articulation_controller = franka.get_articulation_controller()
    # UsdGeom.Imageable(franka.prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    # box = DynamicCuboid(prim_path="/World/box",
    #     position=np.array([0, 0, 1.5]),
    #     scale=np.array([.1, .1, .2]),
    #     color=np.array([.2, .3, 0.]))
    # world.scene.add(box)

    camera = Camera(prim_path="/World/Camera", resolution=(512, 512))
    camera.initialize()

    simulation_time = 0.0
    time_step = world.get_physics_dt()

    tracking_shift = None
    tracker_origin = None
    franka_origin = None
    target = None

    while simulation_app.is_running():
        world.step(render=True)
        rr.set_time_seconds("time", simulation_time)

        with tracker_position_lock:
            if tracker_position['but']:
                tracker_origin = tracker_position['transform']
                franka_origin = Transform3D(*franka.end_effector.get_local_pose())
                tracking_shift = Transform3D(*franka.end_effector.get_local_pose()) * tracker_position['transform'].inverse()
                print(f"tracking_shift: {tracking_shift}")
            elif tracking_shift is not None:
                target = tracking_shift * tracker_position['transform']

                tracking_move = tracker_origin.inverse() * tracker_position['transform']
                franka_move = franka_origin.inverse() * Transform3D(*franka.end_effector.get_local_pose())
                print(f"TD: {tracking_move} FD: {franka_move}")
                log_transform("move/tracking", tracking_move)
                log_transform("move/franka", franka_move)

        log = f"Current: {Transform3D(*franka.end_effector.get_local_pose())}. "
        rr.log("position/current", Transform3D(*franka.end_effector.get_local_pose()).rerun)
        for i, v in zip('xyz', franka.end_effector.get_local_pose()[0]):
            rr.log(f"translation/current/{i}", rr.Scalar(v))
        for i, v in zip('wxyz', franka.end_effector.get_local_pose()[1]):
            rr.log(f"rotation/current/{i}", rr.Scalar(v))

        if target is not None:
            actions, succ = controller.compute_inverse_kinematics(
                target_position=target.position,
                target_orientation=target.orientation.as_quat())  # franka.end_effector.get_local_pose()[1])

            rr.log("position/target", target.rerun)
            for i, v in zip('xyz', target.position):
                rr.log(f"translation/target/{i}", rr.Scalar(v))
            for i, v in zip('wxyz', target.orientation.as_quat()):
                rr.log(f"rotation/target/{i}", rr.Scalar(v))

            if succ:
                articulation_controller.apply_action(actions)
            else:
                log += f"IK failed: target: {target}" # print("IK did not converge to a solution. No action is being taken.")
                print(log)

        for i, v in enumerate(franka.get_joint_positions()):
            rr.log(f"joints/{i}", rr.Scalar(v))

        image = camera.get_rgba()
        if len(image.shape) == 3:
            image = image[:, :, :3]
            rr.log("camera_image", rr.Image(image))

        simulation_time += time_step


if __name__ == "__main__":
    flask_thread = threading.Thread(
        target=app.run, args=('0.0.0.0', 5005), kwargs={'ssl_context': ('cert.pem', 'key.pem')})
    flask_thread.start()

    main()
    flask_thread.join()

simulation_app.close()
rr.shutdown()

