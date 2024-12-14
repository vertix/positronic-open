import asyncio
from collections import namedtuple

import hydra
from lerobot.common.policies.act.modeling_act import ACTPolicy
from omegaconf import DictConfig, OmegaConf
import torch
import rerun as rr

import ironic as ir
from .state import StateEncoder


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


def rerun_log_observation(ts, obs):
    rr.set_time_seconds('time', ts)

    def log_image(name, tensor, compress = True):
        tensor = tensor.squeeze(0)
        tensor = (tensor * 255).type(torch.uint8)
        tensor = tensor.permute(1, 2, 0).cpu().numpy()
        rr_img = rr.Image(tensor)
        if compress:
            rr_img = rr_img.compress()
        rr.log(name, rr_img)

    log_image("observation.images.left", obs['observation.images.left'])
    log_image("observation.images.right", obs['observation.images.right'])
    for i, state in enumerate(obs['observation.state'].squeeze(0)):
        rr.log(f"observation/state/{i}", rr.Scalar(state.item()))

def rerun_log_action(ts, action):
    rr.set_time_seconds('time', ts)
    for i, action in enumerate(action):
        rr.log(f"action/{i}", rr.Scalar(action))



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
            rerun_log_observation(message.timestamp, obs)
            rerun_log_action(message.timestamp, action)

        write_ops = []
        for key, value in action_dict.items():
            out_port = getattr(self.outs, key)
            write_ops.append(out_port.write(ir.Message(value, message.timestamp)))

        await asyncio.gather(*write_ops)
        self.fps.tick()

