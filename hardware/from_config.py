from omegaconf import DictConfig

import pyzed.sl as sl

from hardware.sl_camera import SLCamera

def sl_camera(cfg: DictConfig):
    view = getattr(sl.VIEW, cfg.view)
    resolution = getattr(sl.RESOLUTION, cfg.resolution)
    kwargs = {}
    if 'depth_mode' in cfg:
        kwargs['depth_mode'] = getattr(sl.DEPTH_MODE, cfg.depth_mode)
    if 'coordinate_units' in cfg:
        kwargs['coordinate_units'] = getattr(sl.UNIT, cfg.coordinate_units)
    if 'max_depth' in cfg:
        kwargs['max_depth'] = cfg.max_depth
    if 'depth_mask' in cfg:
        kwargs['depth_mask'] = cfg.depth_mask

    return SLCamera(cfg.fps, view, resolution, **kwargs)
