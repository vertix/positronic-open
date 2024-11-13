import logging
import time
from typing import Optional

import franky

from control import ControlSystem, World, control_system, output_property
from control.utils import FPSCounter
from geom import Transform3D

logger = logging.getLogger(__name__)


@control_system(inputs=["target_position", "gripper_grasped"],
                output_props=["position", "gripper_grasped", "joint_positions", "ext_force_base", "ext_force_ee"])
class Franka(ControlSystem):
    def __init__(self, world: World, ip: str, relative_dynamics_factor: float = 0.2, gripper_speed: float = 0.02,
                 realtime_config: franky.RealtimeConfig = franky.RealtimeConfig.Ignore):
        super().__init__(world)
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
        self.target_fps = FPSCounter("Franka target position")

        try:
            self.gripper = franky.Gripper(ip)
            self.gripper_speed = gripper_speed
            self.gripper_grasped = None
        except Exception as e:
            logger.warning("Did not connect to gripper: %s", e)
            self.gripper = None
            self.gripper_speed = 0.0
            self.gripper_grasped = None

    @output_property('position')
    def position(self):
        """End effector position in robot base coordinate frame."""
        pos = self.robot.current_pose.end_effector_pose
        return Transform3D(pos.translation, pos.quaternion)

    @output_property('joint_positions')
    def joint_positions(self):
        return self.robot.current_joint_state.position

    @output_property('gripper_grasped')
    def gripper_grasped(self):
        if self.gripper:
            return self.gripper_grasped
        return None

    @output_property('ext_force_base')
    def ext_force_base(self):
        state = self.robot.state
        return state.O_F_ext_hat_K

    @output_property('ext_force_ee')
    def ext_force_ee(self):
        state = self.robot.state
        return state.K_F_ext_hat_K

    def on_start(self):
        self.robot.recover_from_errors()
        self.robot.move(franky.JointWaypointMotion([
            franky.JointWaypoint([0.0,  -0.31, 0.0, -1.53, 0.0, 1.522,  0.785])]))

        if self.gripper:
            self.gripper.homing()
            self.gripper_grasped = False

    def on_stop(self):
        print("Franka stopping")
        self.robot.stop()
        if self.gripper:
            self.gripper.open(self.gripper_speed)
        print("Franka stopped")

    def run(self):
        self.on_start()
        with self.ins.subscribe(target_position=self.on_target_position, gripper_grasped=self.on_gripper_grasped):
            try:
                for _ in self.ins.read(): pass
            finally:
                self.on_stop()

    def on_target_position(self, value, _ts):
        try:
            pos = franky.Affine(translation=value.translation, quaternion=value.quaternion)
            self.robot.move(franky.CartesianMotion(pos, franky.ReferenceType.Absolute), asynchronous=True)
            self.target_fps.tick()
        except franky.ControlException as e:
            self.robot.recover_from_errors()
            logger.warning("IK failed for %s: %s", value, e)

    def on_gripper_grasped(self, value, _ts):
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
                    logger.warning("Grasping failed: %s", e)
