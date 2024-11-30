from dataclasses import dataclass, field
from typing import List, Optional

from hydra.core.config_store import ConfigStore
import torch
import torch.nn.functional as F


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
            image = torch.tensor(images[key], dtype=torch.float32).permute(2, 1, 0).unsqueeze(0) / 255
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
