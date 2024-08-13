
import asyncio
import logging
import time

import numpy as np
from control import ControlSystem
from geom import Transform3D, degrees_to_radians, euler_to_quat, quat_to_euler, radians_to_degrees

# This is required because kortex_api relies on the older Python version
import collections
collections.MutableMapping = collections.abc.MutableMapping
collections.MutableSequence = collections.abc.MutableSequence
collections.MutableSet = collections.abc.MutableSet

from kortex_api.autogen.client_stubs.ActuatorConfigClientRpc import ActuatorConfigClient
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, ActuatorConfig_pb2

from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport

from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.messages import Session_pb2

logger = logging.getLogger(__name__)

class Kinova(ControlSystem):
    def __init__(self, ip: str):
        super().__init__(inputs=["target_transform",], outputs=["transform", "joint_positions"])
        self.ip = ip

        self.transport, self.router, self.s_manager = None, None, None
        self.base, self.cyclic = None, None

    async def on_start(self):
        self.transport = TCPTransport()
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)
        self.transport.connect(self.ip, 10000)  # 10000 is the default TCP port for Kinova Gen3

        session_info = Session_pb2.CreateSessionInfo()
        session_info.username = "admin"
        session_info.password = "admin"
        session_info.session_inactivity_timeout = 1000 * 1000   # (milliseconds)
        session_info.connection_inactivity_timeout = 1000 * 1000 # (milliseconds)

        self.s_manager = SessionManager(self.router)
        self.s_manager.CreateSession(session_info)

        self.actuator_config = ActuatorConfigClient(self.router)

        self.base = BaseClient(self.router)
        self.cyclic = BaseCyclicClient(self.router)
        self._frame_id = 0
        self._last_joints = np.zeros(7)

        # This is needed so that robot can be managed by low level control
        self._set_servo_mode(Base_pb2.SINGLE_LEVEL_SERVOING)

        # Move robot to home position
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == "Home":
                action_handle = action.handle
        self.base.ExecuteActionFromReference(action_handle)

        # All other commands are low level
        self._set_servo_mode(Base_pb2.LOW_LEVEL_SERVOING)
        # Set first actuator in torque mode now that the command is equal to measure
        for actuator_id in range(1, 8):
            control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
            control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value('POSITION')
            self.actuator_config.SetControlMode(control_mode_message, actuator_id)

    def _set_servo_mode(self, mode: Base_pb2.ServoingMode):
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = mode
        self.base.SetServoingMode(base_servo_mode)

    async def on_stop(self):
        if self.s_manager != None:
            router_options = RouterClientSendOptions()
            router_options.timeout_ms = 1000
            self.s_manager.CloseSession(router_options)
        self.transport.disconnect()

    @property
    def _current_pose(self):
        """
        Get the tuple of the current pose and joint positions.
        """
        feedback = self.cyclic.RefreshFeedback()
        pos, joints = self._feedback_to_pose(feedback)
        self._last_joints = joints
        return pos, joints

    @classmethod
    def _feedback_to_pose(cls, feedback: BaseCyclic_pb2.Feedback):
        pos =  Transform3D(
            np.array([
                feedback.base.tool_pose_x,
                feedback.base.tool_pose_y,
                feedback.base.tool_pose_z
            ]),
            euler_to_quat(
                degrees_to_radians(
                    np.array([
                        feedback.base.tool_pose_theta_x,
                        feedback.base.tool_pose_theta_y,
                        feedback.base.tool_pose_theta_z
                    ])
                )
            )
        )
        joints = degrees_to_radians(np.array([a.position for a in feedback.actuators]))
        return pos, joints

    def execute_cartesian_move(self, transform: Transform3D):
        euler = radians_to_degrees(quat_to_euler(transform.quaternion))

        ik_data = Base_pb2.IKData()
        ik_data.cartesian_pose.x = transform.translation[0]
        ik_data.cartesian_pose.y = transform.translation[1]
        ik_data.cartesian_pose.z = transform.translation[2]
        ik_data.cartesian_pose.theta_x = euler[0]
        ik_data.cartesian_pose.theta_y = euler[1]
        ik_data.cartesian_pose.theta_z = euler[2]

        for a in radians_to_degrees(self._last_joints):
            ja = ik_data.guess.joint_angles.add()
            ja.value = a

        joint_angels = self.base.ComputeInverseKinematics(ik_data)
        command = BaseCyclic_pb2.Command()
        command.frame_id = self._frame_id % 65536
        for angle in joint_angels.joint_angles:
            ac = command.actuators.add()
            ac.command_id = command.frame_id
            ac.flags = 1  # servoing, whatever that means
            ac.position = angle.value

        send_option = RouterClientSendOptions()
        # send_option.andForget = False
        # send_option.delay_ms = 0
        # send_option.timeout_ms = 3000

        feedback = self.cyclic.Refresh(command, 0, send_option)
        self._frame_id += 1
        return self._feedback_to_pose(feedback)

        # action = Base_pb2.Action()
        # cartesian_pose = action.reach_pose.target_pose
        # cartesian_pose.x = transform.translation[0]
        # cartesian_pose.y = transform.translation[1]
        # cartesian_pose.z = transform.translation[2]
        # cartesian_pose.theta_x = euler[0]
        # cartesian_pose.theta_y = euler[1]
        # cartesian_pose.theta_z = euler[2]
        # self.base.ExecuteAction(action)

    async def run(self):
        await self.on_start()
        try:
            async for input_name, value in self.ins.read(1.0):  # Send current pose at least every second
                if input_name == "target_transform":
                    pos, joints = self.execute_cartesian_move(value)
                else:
                    pos, joints = self._current_pose

                await self.outs.transform.write(pos)
                await self.outs.joint_positions.write(joints)
        finally:
            await self.on_stop()


async def _main():
    kinova = Kinova('192.168.1.10')
    class Manager(ControlSystem):
        def __init__(self):
            super().__init__(inputs=["robot_pos",], outputs=["robot_pos"])

        async def run(self):
            robot_pos = await self.ins.robot_pos.read()
            start_time = time.time()
            # while time.time() - start_time < 60:
            for _ in range(6):
                t = (time.time() - start_time) / 10 * 2 * np.pi      # One period every 10 seconds
                delta = np.array([0., np.cos(t), np.sin(t)]) * 0.1   # Radius of 10cm
                await self.outs.robot_pos.write(Transform3D(robot_pos.translation + delta, robot_pos.quaternion))
                await asyncio.sleep(0.5)

    manager = Manager()
    manager.ins.robot_pos = kinova.outs.transform
    kinova.ins.transform = manager.outs.robot_pos

    await asyncio.gather(kinova.run(), manager.run())


if __name__ == "__main__":
    asyncio.run(_main())