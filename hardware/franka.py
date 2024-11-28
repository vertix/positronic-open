import time
from typing import Optional

import franky
import ironic as ir
from .state import RobotState
from geom import Transform3D


@ir.ironic_system(
    input_ports=['target_position', 'target_grip', 'reset'],
    output_ports=['state'],
    output_props=['position', 'grip', 'joint_positions', 'ext_force_base', 'ext_force_ee']
)
class Franka(ir.ControlSystem):
    def __init__(self, ip: str, relative_dynamics_factor: float = 0.2, gripper_speed: float = 0.02,
                 realtime_config: franky.RealtimeConfig = franky.RealtimeConfig.Ignore):
        super().__init__()
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
        self.target_fps = ir.utils.FPSCounter("Franka target position")
        self._resetting = False

        try:
            self.gripper = franky.Gripper(ip)
            self.gripper_speed = gripper_speed
            self._gripper_grasped = None
        except Exception as e:
            print(f"Did not connect to gripper: {e}")
            self.gripper = None
            self.gripper_speed = 0.0
            self._gripper_grasped = None

    async def setup(self):
        await self.handle_reset(ir.Message(True))

    async def cleanup(self):
        print("Franka stopping")
        self.robot.stop()
        if self.gripper:
            self.gripper.open(self.gripper_speed)
        print("Franka stopped")

    @ir.on_message('target_position')
    async def handle_target_position(self, message: ir.Message):
        try:
            pos = franky.Affine(
                translation=message.data.translation,
                quaternion=message.data.quaternion
            )
            self.robot.move(
                franky.CartesianMotion(pos, franky.ReferenceType.Absolute),
                asynchronous=True
            )
            self.target_fps.tick()
        except franky.ControlException as e:
            self.robot.recover_from_errors()
            print(f"IK failed for {message.data}: {e}")

    @ir.on_message('reset')
    async def handle_reset(self, message: ir.Message):
        """Commands the robot to start moving to the home position."""
        await self.outs.state.write(ir.Message(RobotState.RESETTING, ir.system_clock()))
        self._resetting = True

        self.robot.recover_from_errors()
        self.robot.move(franky.JointWaypointMotion([
            franky.JointWaypoint([0.0, -0.31, 0.0, -1.53, 0.0, 1.522, 0.785])
        ]), asynchronous=True)

        if self.gripper:
            self.gripper.homing()  # This might be a blocking call, so ideally it should run in a separate thread.
            self._gripper_grasped = False


    @ir.on_message('target_grip')
    async def handle_target_grip(self, message: ir.Message):
        if self._gripper_grasped:
            if message.data < 0.33:
                self.gripper.open(self.gripper_speed)
                self._gripper_grasped = False
        else:
            if message.data > 0.66:
                try:
                    self.gripper.grasp(
                        0.0, self.gripper_speed, 20.0,
                        epsilon_outer=1.0 * self.gripper.max_width
                    )
                    self._gripper_grasped = True
                except franky.CommandException as e:
                    print(f"Grasping failed: {e}")

    @ir.out_property
    async def position(self):
        """End effector position in robot base coordinate frame."""
        pos = self.robot.current_pose.end_effector_pose
        return ir.Message(data=Transform3D(pos.translation, pos.quaternion))

    @ir.out_property
    async def joint_positions(self):
        return ir.Message(data=self.robot.current_joint_state.position)

    @ir.out_property
    async def grip(self):
        return ir.Message(
            data=self._gripper_grasped if self.gripper else None,
            timestamp=ir.system_clock()
        )

    @ir.out_property
    async def ext_force_base(self):
        return ir.Message(
            data=self.robot.state.O_F_ext_hat_K,
            timestamp=ir.system_clock()
        )

    @ir.out_property
    async def ext_force_ee(self):
        return ir.Message(
            data=self.robot.state.K_F_ext_hat_K,
            timestamp=ir.system_clock()
        )

    async def step(self) -> ir.State:
        if self._resetting and self.robot.is_in_control():
            self._resetting = False
            await self.outs.state.write(ir.Message(RobotState.AVAILABLE, ir.system_clock()))

        return await super().step()


if __name__ == "__main__":
    import asyncio
    import numpy as np

    async def _main():
        franka = Franka("172.168.0.2")
        await franka.setup()

        target_port = ir.OutputPort('target')
        franka.bind(target_position=target_port)

        current_pos = (await franka.position()).data

        # Move in a small circle
        for t in np.linspace(0, 2*np.pi, 50):
            tr = np.array(current_pos.translation)
            tr[0] += np.cos(t) * 0.1
            tr[1] += np.sin(t) * 0.1
            new_pos = Transform3D(tr, current_pos.quaternion)

            await target_port.write(ir.Message(data=new_pos))
            await asyncio.sleep(0.1)

        await franka.cleanup()

    asyncio.run(_main())
