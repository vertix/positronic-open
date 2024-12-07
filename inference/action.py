from collections import namedtuple

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

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
