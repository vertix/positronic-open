from lerobot.common.policies.act.modeling_act import ACTPolicy

import torch
import numpy as np

from control import ControlSystem, World
from control.utils import FPSCounter
import geom


def _wrap_image(data):
    data = torch.tensor(data)
    data = data.type(torch.float32) / 255
    data = data.permute(2, 0, 1).contiguous()
    data = data.unsqueeze(0)
    return data

class Inference(ControlSystem):
    def __init__(self, world: World, ckpt_path: str, device: str = 'cuda', fps: int = 15):
        super().__init__(world, inputs=['image', 'ext_force_ee', 'ext_force_base',
                                        'robot_position', 'robot_joints', 'grip',
                                        'start', 'stop'],
                                outputs=['target_robot_position', 'target_grip'])
        self.policy_factory = lambda: ACTPolicy.from_pretrained(ckpt_path)
        self.fps = fps
        self.ckpt_path = ckpt_path
        self.device = device

    def _state_tensor(self):
        if self.ins.robot_position.last is None:
            return None

        state = torch.zeros(3 + 4 + 7 + 6 + 6 + 1)  # position, rotation, joint angles, ee force, base force, gripper
        ext_force_ee = self.ins.ext_force_ee.last[1]
        ext_force_base = self.ins.ext_force_base.last[1]
        robot_position = self.ins.robot_position.last[1]
        robot_joints = self.ins.robot_joints.last[1]
        robot_position = self.ins.robot_position.last[1]
        grip = self.ins.grip.last[1] if self.ins.grip.last else 0.

        state[:3] = torch.tensor(robot_position.translation)
        state[3:7] = torch.tensor(robot_position.quaternion)
        state[7:14] = torch.tensor(robot_joints)
        state[14:20] = torch.tensor(ext_force_ee)
        state[20:26] = torch.tensor(ext_force_base)
        state[26] = grip
        return state.unsqueeze(0)

    def run(self):
        policy = self.policy_factory()
        policy.to(self.device)
        running = False
        fps = FPSCounter("Inference")
        for name, ts, data in self.ins.read():
            if name == 'start':
                policy.reset()
                running = True
            elif name == 'stop':
                running = False
            elif running and name == 'image':
                data = _wrap_image(data.image).to(self.device)

                obs = {'observation.images.left': data[:, :, :data.shape[2] // 2, :],
                      'observation.images.right': data[:, :, data.shape[2] // 2:, :],
                      'observation.state': self._state_tensor().to(self.device)}

                action = policy.select_action(obs)
                action = action.squeeze(0).cpu().numpy()

                q = action[3:7]
                q = q / np.linalg.norm(q)
                pos = geom.Transform3D(translation=action[:3], quaternion=q)
                self.outs.target_robot_position.write(pos, ts)
                self.outs.target_grip.write(action[7], ts)
                fps.tick()
