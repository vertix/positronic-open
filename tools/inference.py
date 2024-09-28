from lerobot.common.policies.act.modeling_act import ACTPolicy

from omegaconf import DictConfig
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


class StateEncoder:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def encode_episode(self, episode_data):
        return torch.cat([episode_data[k].unsqueeze(1) if episode_data[k].dim() == 1 else episode_data[k] for k in self.cfg.state], dim=1)

    def encode(self, inputs):
        data = {}
        for k in self.cfg.state:
            k_parts = k.split('.')
            record = inputs[k_parts[0]].last
            if k_parts[0] == 'grip':   # HACK: Get rid of this
                record = None, 0.
            if record is None:
                return None
            tensor = torch.tensor(getattr(record[1], k_parts[1]), dtype=torch.float32) if len(k_parts) > 1 else torch.tensor(record[1], dtype=torch.float32)
            if tensor.ndim == 0:
                tensor = tensor.unsqueeze(0)
            data[k] = tensor

        return torch.cat([data[k] for k in self.cfg.state], dim=0).unsqueeze(0).type(torch.float32)


class Inference(ControlSystem):
    def __init__(self, world: World, cfg: DictConfig):
        # TODO: This list of inputs must be generated from the state config,
        # and the outputs from the action config.
        super().__init__(world, inputs=['image', 'ext_force_ee', 'ext_force_base',
                                        'robot_position', 'robot_joints', 'grip',
                                        'start', 'stop'],
                                outputs=['target_robot_position', 'target_grip'])
        self.policy_factory = lambda: ACTPolicy.from_pretrained(cfg.checkpoint_path)
        self.cfg = cfg

    def _state_tensor(self):
        state = StateEncoder(self.cfg).encode(self.ins)
        if state is None:
            return None
        return state.to(self.cfg.device)

    def run(self):
        policy = self.policy_factory()
        policy.to(self.cfg.device)
        running = False
        fps = FPSCounter("Inference")
        for name, ts, data in self.ins.read():
            if name == 'start':
                policy.reset()
                running = True
            elif name == 'stop':
                running = False
            elif running and name == 'image':
                data = _wrap_image(data.image).to(self.cfg.device)

                state = self._state_tensor()

                obs = {'observation.images.left': data[:, :, :data.shape[2] // 2, :],
                       'observation.images.right': data[:, :, data.shape[2] // 2:, :],
                       'observation.state': state}

                action = policy.select_action(obs)
                action = action.squeeze(0).cpu().numpy()

                q = action[3:7]
                q = q / np.linalg.norm(q)
                pos = geom.Transform3D(translation=action[:3], quaternion=q)
                self.outs.target_robot_position.write(pos, ts)
                self.outs.target_grip.write(action[7], ts)
                fps.tick()
