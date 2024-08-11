import logging

import franky

from control import EventSystem
from geom import Transform

logger = logging.getLogger(__name__)


class Franka(EventSystem):
    def __init__(self, ip: str, relative_dynamics_factor: float = 0.02, gripper_speed: float = 0.02,
                 realtime_config: franky.RealtimeConfig = franky.RealtimeConfig.Enforce):
        super().__init__(
            inputs=["transform", "gripper_grasped"],
            outputs=["transform", "gripper_grasped", "joint_positions"])

        self.robot = franky.Robot(ip, realtime_config=realtime_config)
        self.robot.relative_dynamics_factor = relative_dynamics_factor
        self.robot.set_collision_behavior(
            [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
            [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
            [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
            [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
            [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
            [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
            [30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
            [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
        )

        self.gripper = franky.Gripper(ip)
        self.gripper_speed = gripper_speed
        self.gripper_grasped = None

    def on_start(self):
        self.robot.recover_from_errors()
        motion = franky.JointWaypointMotion([
            franky.JointWaypoint([ 0.0,  -0.3, 0.0, -1.8, 0.0, 1.5,  0.65])])
        self.robot.move(motion)
        self.gripper.homing()
        self.gripper_grasped = False

        pos = self.robot.current_pose.end_effector_pose
        self.outs.transform.write(Transform(pos.translation, pos.quaternion))
        self.outs.joint_positions.write(self.robot.current_joint_state.position)
        self.outs.gripper_grasped.write(self.gripper_grasped)

    def on_stop(self):
        self.robot.stop()
        self.gripper.open(self.gripper_speed)

    @EventSystem.on_event('transform')
    def on_transform(self, value):
        try:
            pos = franky.Affine(translation=value.translation, quaternion=value.quaternion)
            self.robot.move(franky.CartesianMotion(pos, franky.ReferenceType.Absolute), asynchronous=True)
        except franky.ControlException as e:
            self.robot.recover_from_errors()
            logger.warning(f"IK failed for {value}: {e}")

        pos = self.robot.current_pose.end_effector_pose
        self.outs.transform.write(Transform(pos.translation, pos.quaternion))
        self.outs.joint_positions.write(self.robot.current_joint_state.position)

    @EventSystem.on_event('gripper_grasped')
    def on_gripper_grasped(self, value):
        if self.gripper_grasped != value:
            self.gripper_grasped = value
            if self.gripper_grasped:
                try:
                    self.gripper.grasp_async(0.0, self.gripper_speed, 20.0, epsilon_outer=1.0 * self.gripper.max_width)
                except franky.CommandException as e:
                    logger.warning(f"Grasping failed: {e}")
            else:
                self.gripper.open_async(self.gripper_speed)

        self.outs.gripper_grasped.write(self.gripper_grasped)
