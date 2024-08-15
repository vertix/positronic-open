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

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2
from kortex_api.Exceptions.KServerException import KServerException

from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport

from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.messages import Session_pb2

import ruckig

logger = logging.getLogger(__name__)


def _normalise_angles(target, base, period=360):
    res = target - base
    res[res > period / 2] -= period
    res[res < -period / 2] += period
    return base + res

def _feedback_to_pose(feedback: BaseCyclic_pb2.Feedback):
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


class KinovaController:
    def __init__(self, ip: str):
        self.stop_event = threading.Event()
        self.target_joints = None
        self.target_lock = threading.Lock()

        self.output_lock = threading.Lock()
        self.output = None

        self.base = None
        self.last_joints = None

    def inverse_kinematics(self, target_pos):
        euler = radians_to_degrees(quat_to_euler(target_pos.quaternion))
        ik_data = Base_pb2.IKData()
        ik_data.cartesian_pose.x = target_pos.translation[0]
        ik_data.cartesian_pose.y = target_pos.translation[1]
        ik_data.cartesian_pose.z = target_pos.translation[2]
        ik_data.cartesian_pose.theta_x = euler[0]
        ik_data.cartesian_pose.theta_y = euler[1]
        ik_data.cartesian_pose.theta_z = euler[2]

        for a in self.last_joints:
            ja = ik_data.guess.joint_angles.add()
            ja.value = a

        try:
            joint_angels = self.base.ComputeInverseKinematics(ik_data)
            return np.array([a.value for a in joint_angels.joint_angles])
        except KServerException as e:
            logger.error(f"Failed to compute inverse kinematics: {e}")
            return None

    def run(self):
        # region kortex setup
        transport = TCPTransport()
        router = RouterClient(transport, RouterClient.basicErrorCallback)
        transport.connect('192.168.1.10', 10000)

        session_info = Session_pb2.CreateSessionInfo(username="admin", password="admin",
                                                        session_inactivity_timeout=1000 * 1000,
                                                        connection_inactivity_timeout=1000 * 1000)

        sessionManager = SessionManager(router)
        sessionManager.CreateSession(session_info)

        self.base = BaseClient(router)
        self.base.SetServoingMode(Base_pb2.ServoingModeInformation(servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING))

        udp_transport = UDPTransport()
        upd_router = RouterClient(udp_transport, RouterClient.basicErrorCallback)
        udp_transport.connect('192.168.1.10', 10001)

        udp_session = SessionManager(upd_router)
        udp_session.CreateSession(session_info)
        base_cyclic = BaseCyclicClient(upd_router)
        # endregion
        # region home position
        action_list = self.base.ReadAllActions(
            Base_pb2.RequestedActionType(action_type = Base_pb2.REACH_JOINT_ANGLES))
        action_handle = [a.handle for a in action_list.action_list
                        if a.name == "Home"][0]

        def check(n, e):
            if n.action_event in [Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT]:
                print("Home position reached")
                e.set()

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(lambda n: check(n, e), Base_pb2.NotificationOptions())

        self.base.ExecuteActionFromReference(action_handle)
        finished = e.wait(30)
        if not finished:
            raise Exception("Home position not reached")
        self.base.Unsubscribe(notification_handle)
        # endregion

        control = ruckig.RuckigNoThrow(7, 1 / 1000)   # 1000 Hz
        inp_ctrl = ruckig.InputParameter(7)

        inp_ctrl.target_velocity = [0.0] * 7
        inp_ctrl.target_acceleration = [0.0] * 7

        inp_ctrl.max_position = [np.inf, 128.9, np.inf, 147.8, np.inf, 120.3, np.inf]
        inp_ctrl.min_position = [-np.inf, -128.9, -np.inf, -147.8, -np.inf, -120.3, -np.inf]
        inp_ctrl.max_velocity = [50] * 7  # 25 deg / s            # 125
        inp_ctrl.max_acceleration = [57.3] * 4 + [570] * 3
        # inp_ctrl.max_acceleration = [57.3] * 7  # 50 deg / s^2      # 230
        inp_ctrl.max_jerk = [450] * 7   # 6 grad / s^3           # 45000

        target_joints = None
        command = BaseCyclic_pb2.Command()
        for i in range(7):
            act = command.actuators.add()

        send_option = RouterClientSendOptions()
        send_option.timeout_ms = 50

        feedback = base_cyclic.RefreshFeedback()
        self.last_joints = np.array([a.position for a in feedback.actuators])
        inp_ctrl.current_position = self.last_joints
        inp_ctrl.current_velocity = ([a.velocity for a in feedback.actuators])
        with self.output_lock:
            self.output = _feedback_to_pose(feedback)
        out = ruckig.OutputParameter(7)
        res = ruckig.Result.Working
        start = time.monotonic()
        count = 0

        self.base.SetServoingMode(Base_pb2.ServoingModeInformation(servoing_mode=Base_pb2.LOW_LEVEL_SERVOING))
        try:
            while not self.stop_event.is_set():
                with self.target_lock:
                    if self.target_joints is not None:
                        if target_joints is None:
                            start = time.monotonic()

                        target_joints = self.target_joints
                        self.target_joints = None
                        inp_ctrl.target_position = _normalise_angles(target_joints, inp_ctrl.current_position)

                if target_joints is None:
                    time.sleep(0.1)
                    continue

                inp_ctrl.target_position = _normalise_angles(target_joints, inp_ctrl.current_position)

                res = control.update(inp_ctrl, out)
                out.pass_to_input(inp_ctrl)

                command.frame_id = (command.frame_id + 1) % 65536
                for i in range(7):
                    command.actuators[i].command_id = command.frame_id
                    command.actuators[i].position = out.new_position[i]

                feedback = base_cyclic.Refresh(command, 0, send_option)
                pos, joints = _feedback_to_pose(feedback)
                self.last_joints = joints
                with self.output_lock:
                    self.output = pos, joints

                count += 1
                if count % 10000 == 9999:
                    logger.debug(f'Running {count / (time.monotonic() - start):.2f} hz')
                    count, start = 0, time.monotonic()

        finally:
            self.base.SetServoingMode(Base_pb2.ServoingModeInformation(servoing_mode=Base_pb2.SINGLE_LEVEL_SERVOING))
            self.base.Stop()


class Kinova(ControlSystem):
    def __init__(self, ip: str):
        super().__init__(inputs=["target_transform",], outputs=["transform", "joint_positions"])
        self.ip = ip

    async def run(self):
        controller = KinovaController(self.ip)

        pos, joints = None, None
        control_thread = threading.Thread(target=controller.run)
        control_thread.start()
        try:
            async for input_name, value in self.ins.read(1 / 40):
                if input_name == "target_transform":
                    target_joints = controller.inverse_kinematics(value)
                    if target_joints is not None:
                        with controller.target_lock:
                            controller.target_joints = target_joints

                with controller.output_lock:
                    if controller.output is not None:
                        pos, joints = controller.output
                if pos is not None and joints is not None:
                    await self.outs.transform.write(pos)
                    await self.outs.joint_positions.write(joints)
        finally:
            controller.stop_event.set()
            control_thread.join()


async def _main():
    kinova = Kinova('192.168.1.10')
    class Manager(ControlSystem):
        def __init__(self):
            super().__init__(inputs=["robot_pos", "joints"], outputs=["target_pos"])

        async def run(self):
            robot_pos = await self.ins.robot_pos.read()
            start_time = time.time()
            last_command_time = None
            async for input_name, value in self.ins.read(1 / 50):
                if last_command_time is None or time.time() - last_command_time > 0.2:
                    t = (time.time() - start_time) / 8 * 2 * np.pi      # One period every 10 seconds
                    delta = np.array([0., np.cos(t), np.sin(t)]) * 0.25  # Radius of 3 cm
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