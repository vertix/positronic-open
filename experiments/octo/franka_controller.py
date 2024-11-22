from ctypes import c_uint16
import numpy as np
import time
from threading import Lock
import logging
import os

import franky
from pyquaternion import Quaternion
import pymodbus.client as ModbusClient

from widowx_envs.utils.exceptions import Environment_Exception
from widowx_controller.controller_base import RobotControllerBase

class Franka_Controller(RobotControllerBase):
    def __init__(self, robot_name, print_debug, gripper_port="/dev/ttyUSB0",
                 relative_dynamics_factor=0.2, gripper_params=None,
                 enable_rotation='6dof', normal_base_angle=0):
        """
        Args:
            robot_name: IP address of the Franka robot
            print_debug: Enable debug logging
            gripper_port: Serial port for DH gripper (e.g. "/dev/ttyUSB0")
            relative_dynamics_factor: Factor for dynamics (default: 0.2)
            gripper_params: Dict of gripper parameters (unused)
            enable_rotation: Either '6dof' or other (unused)
            normal_base_angle: Base rotation angle (unused)
        """
        super().__init__(robot_name, print_debug)
        print('Initializing Franka controller...')

        # Initialize robot
        self.robot = franky.Robot(robot_name, realtime_config=franky.RealtimeConfig.Ignore)
        self.robot.relative_dynamics_factor = relative_dynamics_factor
        self._setup_collision_behavior()

        # Initialize gripper
        self.gripper = ModbusClient.ModbusSerialClient(
            port=gripper_port,
            baudrate=115200,
            bytesize=8,
            parity="N",
            stopbits=1
        )

        # Setup logging
        logger = logging.getLogger('robot_logger')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_level = logging.DEBUG if print_debug else logging.WARN
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Initialize state tracking
        self._joint_lock = Lock()
        self._gripper_state = 0.0  # 0 = open, 1 = closed
        self._moving_time = 2.0

        # Initialize robot
        self.robot.recover_from_errors()
        self._move_to_default_pose()
        self._init_gripper()

    def _setup_collision_behavior(self):
        """Configure collision detection thresholds"""
        self.robot.set_collision_behavior(
            [50.0] * 7,  # lower_torque_thresholds_acceleration
            [50.0] * 7,  # upper_torque_thresholds_acceleration
            [30.0] * 7,  # lower_torque_thresholds_nominal
            [30.0] * 7,  # upper_torque_thresholds_nominal
            [50.0] * 6,  # lower_force_thresholds_acceleration
            [50.0] * 6,  # upper_force_thresholds_acceleration
            [30.0] * 6,  # lower_force_thresholds_nominal
            [30.0] * 6   # upper_force_thresholds_nominal
        )

    def _move_to_default_pose(self):
        """Move robot to default joint configuration"""
        self.robot.move(franky.JointWaypointMotion([
            franky.JointWaypoint([0.0, -0.31, 0.0, -1.53, 0.0, 1.522, 0.785])
        ]))

    def _init_gripper(self):
        """Initialize the DH gripper"""
        if self._gripper_state_g() != 1 or self._gripper_state_r() != 1:
            self.gripper.write_register(0x100, 0xa5, slave=1)
            while self._gripper_state_g() != 1 and self._gripper_state_r() != 1:
                time.sleep(0.1)

        # Set initial gripper parameters
        self.gripper.write_register(0x101, c_uint16(100).value, slave=1)  # force
        self.gripper.write_register(0x104, c_uint16(100).value, slave=1)  # speed
        self.open_gripper()
        time.sleep(0.5)

    def _gripper_state_g(self):
        return self.gripper.read_holding_registers(0x200, 1, slave=1).registers[0]

    def _gripper_state_r(self):
        return self.gripper.read_holding_registers(0x20A, 1, slave=1).registers[0]

    def set_moving_time(self, moving_time):
        """Set the moving time for trajectories"""
        self._moving_time = moving_time

    def move_to_state(self, target_xyz, target_zangle, duration=1.5):
        """Move end-effector to target position and orientation"""
        current_pose = self.robot.current_pose.end_effector_pose
        new_quat = Quaternion(axis=np.array([0.0, 0.0, 1.0]), angle=target_zangle)

        target_pose = franky.Affine(
            translation=target_xyz,
            quaternion=new_quat.elements
        )

        try:
            self.robot.move(
                franky.CartesianMotion(target_pose, franky.ReferenceType.Absolute),
                asynchronous=False
            )
        except franky.ControlException as e:
            print(f"Move failed: {e}")
            raise Environment_Exception

    def move_to_eep(self, target_pose, duration=1.5, blocking=True, check_effort=True, step=True):
        """Move to end-effector pose matrix"""
        try:
            franka_pose = franky.Affine(
                translation=target_pose[:3, 3],
                rotation=target_pose[:3, :3]
            )
            self.robot.move(
                franky.CartesianMotion(franka_pose, franky.ReferenceType.Absolute),
                asynchronous=not blocking
            )
        except franky.ControlException as e:
            print(f"Move failed: {e}")
            raise Environment_Exception

    def set_joint_angles(self, target_positions, duration=4):
        """Move to target joint positions"""
        try:
            self.robot.move(franky.JointWaypointMotion([
                franky.JointWaypoint(target_positions)
            ]))
        except franky.ControlException as e:
            print(f"Joint move failed: {e}")
            raise Environment_Exception

    def move_to_neutral(self, duration=4):
        """Move robot to neutral pose"""
        print('Moving to neutral position...')
        try:
            self._move_to_default_pose()
        except franky.ControlException as e:
            print(f"Failed to move to neutral: {e}")
            self.robot.recover_from_errors()
            raise Environment_Exception

    def get_cartesian_pose(self, matrix=False):
        """Get current end-effector pose"""
        pose = self.robot.current_pose.end_effector_pose
        if matrix:
            return pose.matrix
        else:
            return np.concatenate([pose.translation, pose.quaternion])

    def get_joint_angles(self):
        """Get current joint angles"""
        return np.array(self.robot.current_joint_state.position)

    def get_joint_angles_velocity(self):
        """Get current joint velocities"""
        return np.array(self.robot.current_joint_state.velocity)

    def get_joint_effort(self):
        """Get current joint efforts/torques"""
        return np.array(self.robot.current_joint_state.effort)

    def open_gripper(self, wait=False):
        """Open the gripper"""
        self.gripper.write_register(0x103, c_uint16(1000).value, slave=1)
        self._gripper_state = 0.0
        if wait:
            time.sleep(1.0)

    def close_gripper(self, wait=False):
        """Close the gripper"""
        self.gripper.write_register(0x103, c_uint16(0).value, slave=1)
        self._gripper_state = 1.0
        if wait:
            time.sleep(1.0)

    def get_gripper_position(self):
        """Get current gripper position"""
        response = self.gripper.read_holding_registers(0x202, 1, slave=1)
        if response.isError():
            raise Exception(f"Error reading gripper position: {response}")
        return 1 - response.registers[0] / 1000

    def get_gripper_desired_position(self):
        """Get target gripper position"""
        return self._gripper_state

    def clean_shutdown(self):
        """Clean shutdown of the robot"""
        pid = os.getpid()
        logging.getLogger('robot_logger').info('Shutting down Franka controller w/ pid: {}'.format(pid))
        self.open_gripper()
        self.move_to_neutral()
        logging.shutdown()
        os.kill(pid, 9)