import logging
import time
from typing import Optional

import franky

from control import ControlSystem, World
from control.utils import FPSCounter
from geom import Transform3D

logger = logging.getLogger(__name__)


class Franka(ControlSystem):
    def __init__(self, world: World, ip: str, relative_dynamics_factor: float = 0.2, gripper_speed: float = 0.02,
                 reporting_frequency: Optional[float] = None,
                 realtime_config: franky.RealtimeConfig = franky.RealtimeConfig.Ignore):
        """
        Args:
            reporting_frequency: Frequency at which to report outputs. If None, they will be reported only on inputs.
        """
        super().__init__(
            world,
            inputs=["target_position", "gripper_grasped"],
            outputs=["position", "gripper_grasped", "joint_positions", "ext_force_base", "ext_force_ee"])

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
        self.reporting_frequency = reporting_frequency

        try:
            self.gripper = franky.Gripper(ip)
            self.gripper_speed = gripper_speed
            self.gripper_grasped = None
        except Exception as e:
            logger.warning(f"Did not connect to gripper: {e}")
            self.gripper = None
            self.gripper_speed = 0.0
            self.gripper_grasped = None

    def _write_outputs(self):
        pos, joints, state = self.robot.current_pose.end_effector_pose, self.robot.current_joint_state, self.robot.state
        ts = int(time.time() * 1000)

        self.outs.position.write(Transform3D(pos.translation, pos.quaternion), ts)
        self.outs.joint_positions.write(joints.position, ts)
        if self.gripper:
            self.outs.gripper_grasped.write(self.gripper_grasped, ts)

        if self.outs.ext_force_base.subscribed:
            self.outs.ext_force_base.write(state.O_F_ext_hat_K, ts)
        if self.outs.ext_force_ee.subscribed:
            self.outs.ext_force_ee.write(state.K_F_ext_hat_K, ts)

    def on_start(self):
        self.robot.recover_from_errors()
        self.robot.move(franky.JointWaypointMotion([
            franky.JointWaypoint([0.0,  -0.31, 0.0, -1.53, 0.0, 1.522,  0.785])]))

        if self.gripper:
            self.gripper.homing()
            self.gripper_grasped = False

        self._write_outputs()

    def on_stop(self):
        print("Franka stopping")
        self.robot.stop()
        if self.gripper:
            self.gripper.open(self.gripper_speed)
        print("Franka stopped")

    def run(self):
        self.on_start()
        try:
            to = 1.0 / self.reporting_frequency if self.reporting_frequency is not None else None
            fps = FPSCounter("Franka")
            for name, _ts, value in self.ins.read(timeout=to):
                if name == "target_position":
                    self.on_target_position(value)
                elif name == "gripper_grasped":
                    self.on_gripper_grasped(value)

                self._write_outputs()
                fps.tick()
        finally:
            self.on_stop()

    def on_target_position(self, value):
        try:
            pos = franky.Affine(translation=value.translation, quaternion=value.quaternion)
            self.robot.move(franky.CartesianMotion(pos, franky.ReferenceType.Absolute), asynchronous=True)
        except franky.ControlException as e:
            self.robot.recover_from_errors()
            logger.warning(f"IK failed for {value}: {e}")

    def on_gripper_grasped(self, value):
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
