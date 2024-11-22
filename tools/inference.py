import asyncio
from collections import namedtuple
from dataclasses import dataclass, field
from typing import List, Optional

import hydra
from hydra.core.config_store import ConfigStore
from lerobot.common.policies.act.modeling_act import ACTPolicy
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
import rerun as rr

import ironic as ir

@dataclass
class ImageEncodingConfig:
    key: str = "image"
    resize: Optional[List[int]] = None

@dataclass
class StateEncodingConfig:
    images: List[ImageEncodingConfig] = field(default_factory=list)
    resize: Optional[List[int]] = None
    state: List[str] = field(default_factory=list)

ConfigStore.instance().store(name="state", node=StateEncodingConfig)


class StateEncoder:
    def __init__(self, cfg: StateEncodingConfig):
        self.cfg = cfg

    def encode_episode(self, episode_data):
        obs = {}
        for cfg in self.cfg.images:
            image = episode_data['image.' + cfg.key]
            image = image.permute(0, 3, 2, 1)
            if cfg.resize is not None:
                # We use nearest because we mostly have downscaling
                image = F.interpolate(image, size=tuple(cfg.resize), mode='nearest')

            obs[f"observation.images.{cfg.key}"] = image.permute(0, 1, 3, 2)

        obs['observation.state'] = torch.cat([episode_data[k].unsqueeze(1) if episode_data[k].dim() == 1 else episode_data[k]
                                              for k in self.cfg.state], dim=1)
        return obs

    def encode(self, images, inputs):
        obs = {}
        for key in images:
            side = key.split('_')[-1]
            image = torch.tensor(images[key], dtype=torch.float32).permute(2, 1, 0).contiguous().unsqueeze(0) / 255
            if self.cfg.resize is not None:
                # Bilinear might be better, but due to NaNs in depth, we use nearest
                image = F.interpolate(image, size=tuple(self.cfg.resize), mode='nearest')
            obs[f"observation.images.{side}"] = image.permute(0, 1, 3, 2)

        data = {}
        for key in self.cfg.state:
            tensor = torch.tensor(inputs[key], dtype=torch.float32)
            if tensor.ndim == 0:
                tensor = tensor.unsqueeze(0)
            data[key] = tensor

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

    def decode(self, action_vector, inputs):
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
                return inputs[cfg.name]
            else:
                return cfg

        cfg_copy = OmegaConf.to_container(self.cfg, resolve=False)
        cfg_copy = OmegaConf.create(cfg_copy)
        cfg = replace_fields(cfg_copy.outputs)
        outputs = hydra.utils.instantiate(cfg)
        return outputs


@ir.ironic_system(
    input_ports=['frame', 'start', 'stop'],
    input_props=['robot_data'],
    output_ports=['target_robot_position', 'target_grip']
)
class Inference(ir.ControlSystem):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.policy_factory = lambda: ACTPolicy.from_pretrained(cfg.checkpoint_path)
        self.cfg = cfg
        self.state_encoder = StateEncoder(hydra.utils.instantiate(cfg.state))
        self.action_decoder = ActionDecoder(cfg.action)
        self.policy = None
        self.running = False
        self.fps = ir.utils.FPSCounter("Inference")

    async def setup(self):
        """Initialize the policy"""
        self.policy = self.policy_factory()
        self.policy.to(self.cfg.device)

    @ir.on_message("start")
    async def handle_start(self, message: ir.Message):
        """Handle policy start message"""
        self.policy.reset()
        self.fps.reset()
        self.running = True

    @ir.on_message("stop")
    async def handle_stop(self, message: ir.Message):
        """Handle policy stop message"""
        self.fps.report()
        self.running = False

    @ir.on_message("frame")
    async def handle_frame(self, message: ir.Message):
        """Process image and generate actions"""
        if not self.running:
            return

        robot_data = (await self.ins.robot_data()).data
        obs = self.state_encoder.encode(message.data, robot_data)
        for key in obs:
            obs[key] = obs[key].to(self.cfg.device)

        action = self.policy.select_action(obs).squeeze(0).cpu().numpy()
        action_dict = self.action_decoder.decode(action, robot_data)

        if self.cfg.rerun:
            self._log_observation(message.timestamp, obs)
            self._log_action(message.timestamp, action)

        write_ops = []
        for key, value in action_dict.items():
            out_port = getattr(self.outs, key)
            write_ops.append(out_port.write(ir.Message(value, message.timestamp)))

        await asyncio.gather(*write_ops)
        self.fps.tick()

    def _log_observation(self, ts, obs):
        rr.set_time_seconds('time', ts / 1000)

        def log_image(name, tensor):
            tensor = tensor.squeeze(0)
            tensor = (tensor * 255).type(torch.uint8)
            tensor = tensor.permute(1, 2, 0).cpu().numpy()
            rr.log(name, rr.Image(tensor))

        log_image("observation.images.left", obs['observation.images.left'])
        log_image("observation.images.right", obs['observation.images.right'])
        for i, state in enumerate(obs['observation.state'].squeeze(0)):
            rr.log(f"observation/state/{i}", rr.Scalar(state.item()))

    def _log_action(self, ts, action):
        rr.set_time_seconds('time', ts / 1000)
        for i, action in enumerate(action):
            rr.log(f"action/{i}", rr.Scalar(action))
