
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

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport

from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.messages import Session_pb2

logger = logging.getLogger(__name__)

class Kinova(ControlSystem):
    def __init__(self, ip: str):
        super().__init__(inputs=["transform",], outputs=["transform", "joint_positions"])
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

        self.base = BaseClient(self.router)
        self.cyclic = BaseCyclicClient(self.router)

        # TODO: Not sure if this is needed
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        # Move robot to home position
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == "Home":
                action_handle = action.handle
        self.base.ExecuteActionFromReference(action_handle)

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
        action = Base_pb2.Action()
        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = transform.translation[0]
        cartesian_pose.y = transform.translation[1]
        cartesian_pose.z = transform.translation[2]
        euler = radians_to_degrees(quat_to_euler(transform.quaternion))
        cartesian_pose.theta_x = euler[0]
        cartesian_pose.theta_y = euler[1]
        cartesian_pose.theta_z = euler[2]
        self.base.ExecuteAction(action)

    async def run(self):
        await self.on_start()
        try:
            async for input_name, value in self.ins.read(1.0):  # Send current pose at least every second
                if input_name == "transform":
                    self.execute_cartesian_move(value)

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
            while time.time() - start_time < 60:
                t = (time.time() - start_time) / 10 * 2 * np.pi # One period every 10 seconds
                delta = np.array([0., np.cos(t), np.sin(t)]) * 0.1  # Radius of 10cm
                await self.outs.robot_pos.write(Transform3D(robot_pos.translation + delta, robot_pos.quaternion))
                await asyncio.sleep(0.1)

    manager = Manager()
    manager.ins.robot_pos = kinova.outs.transform
    kinova.ins.transform = manager.outs.robot_pos

    await asyncio.gather(kinova.run(), manager.run())


if __name__ == "__main__":
    asyncio.run(_main())