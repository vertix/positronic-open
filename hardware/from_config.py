from omegaconf import DictConfig

import pyzed.sl as sl

from hardware.sl_camera import SLCamera

def sl_camera(cfg: DictConfig):
    view = getattr(sl.VIEW, cfg.view)
    resolution = getattr(sl.RESOLUTION, cfg.resolution)
    depth_mode = getattr(sl.DEPTH_MODE, cfg.depth_mode) if 'depth_mode' in cfg else sl.DEPTH_MODE.NONE
    coordinate_units = getattr(sl.UNIT, cfg.coordinate_units) if 'coordinate_units' in cfg else sl.UNIT.METER

    return SLCamera(cfg.fps, view, resolution, depth_mode, coordinate_units)
