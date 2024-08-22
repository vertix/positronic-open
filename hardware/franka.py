import logging

import franky

from control import EventSystem
from geom import Transform3D

logger = logging.getLogger(__name__)


class Franka(EventSystem):
    def __init__(self, ip: str, relative_dynamics_factor: float = 0.02, gripper_speed: float = 0.02,
                 realtime_config: franky.RealtimeConfig = franky.RealtimeConfig.Ignore):
        super().__init__(
            inputs=["target_position", "gripper_grasped"],
            outputs=["position", "gripper_grasped", "joint_positions"])

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

        try:
            self.gripper = franky.Gripper(ip)
            self.gripper_speed = gripper_speed
            self.gripper_grasped = None
        except Exception as e:
            logger.warning(f"Did not connect to gripper: {e}")
            self.gripper = None
            self.gripper_speed = 0.0
            self.gripper_grasped = None

    async def on_start(self):
        self.robot.recover_from_errors()
        self.robot.move(franky.JointWaypointMotion([
            franky.JointWaypoint([0.0,  -0.31, 0.0, -1.83, 0.0, 1.522,  0.785])]))

        if self.gripper:
            self.gripper.homing()
            self.gripper_grasped = False

        pos = self.robot.current_pose.end_effector_pose
        await self.outs.position.write(Transform3D(pos.translation, pos.quaternion))
        await self.outs.joint_positions.write(self.robot.current_joint_state.position)
        await self.outs.gripper_grasped.write(self.gripper_grasped)

    async def on_stop(self):
        self.robot.stop()
        if self.gripper:
            self.gripper.open(self.gripper_speed)

    @EventSystem.on_event('target_position')
    async def on_target_position(self, value):
        try:
            pos = franky.Affine(translation=value.translation, quaternion=value.quaternion)
            self.robot.move(franky.CartesianMotion(pos, franky.ReferenceType.Absolute), asynchronous=True)
        except franky.ControlException as e:
            self.robot.recover_from_errors()
            logger.warning(f"IK failed for {value}: {e}")

        pos = self.robot.current_pose.end_effector_pose
        await self.outs.position.write(Transform3D(pos.translation, pos.quaternion))
        await self.outs.joint_positions.write(self.robot.current_joint_state.position)

    @EventSystem.on_event('gripper_grasped')
    async def on_gripper_grasped(self, value):
        if self.gripper_grasped:
            if value < 0.33:
                self.gripper.open(self.gripper_speed)
                self.gripper_grasped = False
        else:
            if value > 0.66:
                try:
                    self.gripper.grasp(0.0, self.gripper_speed, 20.0,
                                       epsilon_outer=1.0 * self.gripper.max_width)
                    self.gripper_grasped = True
                except franky.CommandException as e:
                    logger.warning(f"Grasping failed: {e}")

        await self.outs.gripper_grasped.write(self.gripper_grasped)
