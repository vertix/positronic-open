from collections import namedtuple
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import hydra
from hydra.core.config_store import ConfigStore
from lerobot.common.policies.act.modeling_act import ACTPolicy
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
import numpy as np

from control import ControlSystem, World
from control.utils import FPSCounter
import geom


@dataclass
class ImageEncodingConfig:
    key: str = "image"
    resize: Optional[List[int]] = None

@dataclass
class StateEncodingConfig:
    left: ImageEncodingConfig = field(default_factory=ImageEncodingConfig)
    right: ImageEncodingConfig = field(default_factory=ImageEncodingConfig)
    state: List[str] = field(default_factory=list)

ConfigStore.instance().store(name="state", node=StateEncodingConfig)


class StateEncoder:
    def __init__(self, cfg: StateEncodingConfig):
        self.cfg = cfg

    def encode_episode(self, episode_data):
        obs = {}
        image = episode_data['image']
        image = image.permute(0, 3, 2, 1)
        left = image[:, :, :image.shape[2] // 2, :]
        right = image[:, :, image.shape[2] // 2:, :]
        if self.cfg.left.resize is not None:
            left = F.interpolate(left, size=tuple(self.cfg.left.resize), mode='bilinear')
        if self.cfg.right.resize is not None:
            right = F.interpolate(right, size=tuple(self.cfg.right.resize), mode='bilinear')
        obs['observation.images.left'] = left.permute(0, 1, 3, 2)
        obs['observation.images.right'] = right.permute(0, 1, 3, 2)
        obs['observation.state'] = torch.cat([episode_data[k].unsqueeze(1) if episode_data[k].dim() == 1 else episode_data[k]
                                              for k in self.cfg.state], dim=1)
        return obs

    def encode(self, image, inputs):
        obs = {}
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).contiguous().unsqueeze(0) / 255
        left = image[:, :, :image.shape[2] // 2, :]
        right = image[:, :, image.shape[2] // 2:, :]
        if self.cfg.left.resize is not None:
            left = F.interpolate(left, size=tuple(self.cfg.left.resize), mode='bilinear')
        if self.cfg.right.resize is not None:
            right = F.interpolate(right, size=tuple(self.cfg.right.resize), mode='bilinear')
        obs['observation.images.left'] = left
        obs['observation.images.right'] = right

        data = {}
        for k in self.cfg.state:
            k_parts = k.split('.')
            # Record is a tuple of (timestamp, data) or None if no data is available
            record = inputs[k_parts[0]].last
            # We consider grip to be zero in case we did not receive any grip data yet
            if k_parts[0] == 'grip' and record is None:   # HACK: Get rid of this
                record = None, 0.
            if record is None:
                return None
            tensor = torch.tensor(getattr(record[1], k_parts[1]), dtype=torch.float32) if len(k_parts) > 1 else torch.tensor(record[1], dtype=torch.float32)
            if tensor.ndim == 0:
                tensor = tensor.unsqueeze(0)
            data[k] = tensor

        obs['observation.state'] = torch.cat([data[k] for k in self.cfg.state], dim=0).unsqueeze(0).type(torch.float32)
        return obs


Field = namedtuple('Field', ['name'])
InputField = namedtuple('InputField', ['name'])
OmegaConf.register_new_resolver('field', lambda x: Field(x))
OmegaConf.register_new_resolver('input', lambda x: InputField(x))

class ActionDecoder:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def encode_episode(self, episode_data):
        def replace_inputs(cfg):
            if OmegaConf.is_dict(cfg):
                return {k: replace_inputs(v) for k, v in cfg.items()}
            elif OmegaConf.is_list(cfg):
                return [replace_inputs(item) for item in cfg]
            elif isinstance(cfg, InputField):
                return episode_data[cfg.name]
            else:
                return cfg

        cfg_copy = OmegaConf.to_container(self.cfg, resolve=False)
        cfg_copy = OmegaConf.create(cfg_copy)
        cfg = replace_inputs(cfg_copy.fields)
        cfg = hydra.utils.instantiate(cfg)

        records = [record.value.unsqueeze(1) if record.value.dim() == 1 else record.value for record in cfg.values()]
        return torch.cat(records, dim=1)

    def decode(self, action_vector, input_ports):
        start = 0
        fields = {}
        for name in self.cfg.fields:
            fields[name] = action_vector[start:start + self.cfg.fields[name].length]
            start += self.cfg.fields[name].length

        def replace_fields(cfg):
            if OmegaConf.is_dict(cfg):
                return {k: replace_fields(v) for k, v in cfg.items()}
            elif OmegaConf.is_list(cfg):
                return [replace_fields(item) for item in cfg]
            elif isinstance(cfg, Field):
                return fields[cfg.name]
            elif isinstance(cfg, InputField):
                key, field = (cfg.name.split('.') + [None])[:2]
                record = input_ports[key].last
                if record is None:
                    raise ValueError(f"Input field {cfg.name} does not have a value")
                return getattr(record[1], field) if field else record[1]
            else:
                return cfg

        cfg_copy = OmegaConf.to_container(self.cfg, resolve=False)
        cfg_copy = OmegaConf.create(cfg_copy)
        cfg = replace_fields(cfg_copy.outputs)
        outputs = hydra.utils.instantiate(cfg)
        return outputs


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
        self.state_encoder = StateEncoder(hydra.utils.instantiate(cfg.state))
        self.action_decoder = ActionDecoder(cfg.action)

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
                obs = self.state_encoder.encode(data.image, self.ins)
                for key in obs:
                    obs[key] = obs[key].to(self.cfg.device)

                action = policy.select_action(obs)
                action_dict = self.action_decoder.decode(action.squeeze(0).cpu().numpy(), self.ins)
                for key in action_dict:
                    self.outs[key].write(action_dict[key], ts)
                fps.tick()
