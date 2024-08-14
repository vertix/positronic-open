import asyncio
import logging
import threading
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

import ruckig
import rerun as rr
rr.init("kinova", spawn=False)
rr.save('kinova.rrd')

logger = logging.getLogger(__name__)

class Kinova(ControlSystem):
    def __init__(self, ip: str):
        super().__init__(inputs=["target_transform",], outputs=["transform", "joint_positions"])
        self.ip = ip

        self.transport, self.router, self.s_manager = None, None, None
        self.base, self.cyclic = None, None

    def on_start(self):
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
        self._last_joints = np.zeros(7)

        # This is needed so that robot can be managed by low level control
        self.base.SetServoingMode(Base_pb2.ServoingModeInformation(servoing_mode=Base_pb2.SINGLE_LEVEL_SERVOING))

        # Move robot to home position
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = [a.handle for a in action_list.action_list if a.name == "Home"][0]

        def check(n, e):
            if n.action_event in [Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT]:
                e.set()

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            lambda n: check(n, e), Base_pb2.NotificationOptions())

        self.base.ExecuteActionFromReference(action_handle)
        finished = e.wait(30)
        if not finished:
            raise Exception("Home position not reached")
        self.base.Unsubscribe(notification_handle)
        print("Home position reached")

    def on_stop(self):
        self.base.Stop()
        if self.s_manager != None:
            router_options = RouterClientSendOptions()
            router_options.timeout_ms = 1000
            self.s_manager.CloseSession(router_options)
        self.transport.disconnect()
        print("Disconnected")

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
        joints = np.array([a.position for a in feedback.actuators])
        return pos, joints

    def _inverse_kinematics(self, transform: Transform3D, joints_guess: np.ndarray):
        euler = radians_to_degrees(quat_to_euler(transform.quaternion))
        ik_data = Base_pb2.IKData()
        ik_data.cartesian_pose.x = transform.translation[0]
        ik_data.cartesian_pose.y = transform.translation[1]
        ik_data.cartesian_pose.z = transform.translation[2]
        ik_data.cartesian_pose.theta_x = euler[0]
        ik_data.cartesian_pose.theta_y = euler[1]
        ik_data.cartesian_pose.theta_z = euler[2]

        for a in joints_guess:
            ja = ik_data.guess.joint_angles.add()
            ja.value = a

        joint_angels = self.base.ComputeInverseKinematics(ik_data)
        return np.array([a.value for a in joint_angels.joint_angles])

    @classmethod
    def _normalise_angles(cls, target, base, period=360):
        res = target - base
        res[res > period / 2] -= period
        res[res < -period / 2] += period
        return base + res

    async def _run(self):
        self.on_start()
        try:
            last_joints = None
            async for input_name, value in self.ins.read(1 / 40):  # 40 Hz
                feedback = self.cyclic.RefreshFeedback()
                pos, joints = self._feedback_to_pose(feedback)
                await self.outs.transform.write(pos)

                if last_joints is not None:
                    joints = self._normalise_angles(joints, last_joints)
                last_joints = joints
                await self.outs.joint_positions.write(degrees_to_radians(joints))

                if input_name == "target_transform":
                    action = Base_pb2.Action()
                    cartesian_pose = action.reach_pose.target_pose
                    cartesian_pose.x = value.translation[0]
                    cartesian_pose.y = value.translation[1]
                    cartesian_pose.z = value.translation[2]
                    euler = radians_to_degrees(quat_to_euler(value.quaternion))
                    cartesian_pose.theta_x = euler[0]
                    cartesian_pose.theta_y = euler[1]
                    cartesian_pose.theta_z = euler[2]
                    self.base.ExecuteAction(action)
        except KeyboardInterrupt:
            print("Stopping Kinova ...")
        finally:
            self.on_stop()

    async def run(self):
        self.on_start()
        try:
            control = ruckig.Ruckig(7, 1 / 20)  # 20 Hz
            tracking = False
            inp_ctrl = ruckig.InputParameter(7)

            inp_ctrl.target_velocity = [0.0] * 7
            inp_ctrl.target_acceleration = [0.0] * 7

            inp_ctrl.max_position = [np.inf, 128.9, np.inf, 147.8, np.inf, 120.3, np.inf]
            inp_ctrl.min_position = [-np.inf, -128.9, -np.inf, -147.8, -np.inf, -120.3, -np.inf]
            inp_ctrl.max_velocity = [30.0] * 7
            inp_ctrl.max_acceleration = [15.0] * 7
            inp_ctrl.max_jerk = [6.0] * 7

            out = ruckig.OutputParameter(7)
            last_joints = None
            async for input_name, value in self.ins.read(1 / 40):  # 40 Hz
                feedback = self.cyclic.RefreshFeedback()
                pos, joints = self._feedback_to_pose(feedback)
                await self.outs.transform.write(pos)

                if last_joints is not None:
                    joints = self._normalise_angles(joints, last_joints)
                last_joints = joints
                await self.outs.joint_positions.write(degrees_to_radians(joints))

                if input_name == "target_transform":
                    target_joints = self._inverse_kinematics(value, joints)
                    inp_ctrl.target_position = self._normalise_angles(target_joints, joints)
                    tracking = True

                if tracking:
                    inp_ctrl.current_position = joints
                    inp_ctrl.target_position = self._normalise_angles(inp_ctrl.target_position, joints)

                    _res = control.update(inp_ctrl, out)
                    out.pass_to_input(inp_ctrl)

                    joint_speeds = Base_pb2.JointSpeeds()
                    for i in range(7):
                        js = joint_speeds.joint_speeds.add()
                        js.joint_identifier = i
                        js.value = out.new_velocity[i]
                        js.duration = 2  # Some caution period

                    self.base.SendJointSpeedsCommand(joint_speeds)

        except KeyboardInterrupt:
            print("Stopping Kinova ...")
        finally:
            self.on_stop()


async def _main():
    kinova = Kinova('192.168.1.10')
    class Manager(ControlSystem):
        def __init__(self):
            super().__init__(inputs=["robot_pos", "joints"], outputs=["target_pos"])

        async def run(self):
            robot_pos = await self.ins.robot_pos.read()
            start_time = time.time()
            last_command_time = None
            async for input_name, value in self.ins.read(1 / 40):
                if input_name == "joints":
                    rr.set_time_seconds('time', time.time() - start_time)
                    for i, v in enumerate(radians_to_degrees(value)):
                        rr.log(f"joint_{i}", rr.Scalar(v))

                if last_command_time is None or time.time() - last_command_time > 1:
                    t = (time.time() - start_time) / 10 * 2 * np.pi      # One period every 10 seconds
                    delta = np.array([0., np.cos(t), np.sin(t)]) * 0.20   # Radius of 10 cm
                    target = Transform3D(robot_pos.translation + delta, robot_pos.quaternion)
                    last_command_time = time.time()
                    await self.outs.target_pos.write(target)

                if time.time() - start_time > 30:
                    break


    manager = Manager()
    manager.ins.robot_pos = kinova.outs.transform
    manager.ins.joints = kinova.outs.joint_positions
    kinova.ins.target_transform = manager.outs.target_pos

    await asyncio.gather(kinova.run(), manager.run())


if __name__ == "__main__":
    asyncio.run(_main())