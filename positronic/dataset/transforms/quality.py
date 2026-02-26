"""Per-frame quality signals and scalar aggregators for dataset debugging.

Quality signals are composed from general-purpose signal transforms (diff, norm,
view) — nothing expensive happens until values are accessed. Scalar aggregators
extract all values from a signal and compute a summary statistic.
"""

import numpy as np

from .signals import Elementwise, Join, diff, norm, view

_TRANSLATION = slice(0, 3)
_DT_SEC = 1 / 15


def idle_mask(episode, signal='robot_state.q', velocity_threshold=0.015, dt_sec=_DT_SEC):
    """Per-frame bool: True where joint speed < threshold (rad/s)."""
    speed = norm(diff(episode.signals[signal], dt_sec))

    def fn(vals):
        return np.array(vals) < velocity_threshold

    return Elementwise(speed, fn)


def jerk(episode, signal='robot_state.q', dt_sec=_DT_SEC):
    """Per-frame joint acceleration magnitude (rad/s^2)."""
    return norm(diff(episode.signals[signal], dt_sec, order=2))


def cmd_lag(episode, cmd_signal='robot_commands.pose', state_signal='robot_state.ee_pose', components=_TRANSLATION):
    """Per-frame distance between commanded and actual pose (meters)."""
    cmd = episode.signals[cmd_signal]
    ee = episode.signals[state_signal]

    def fn(pairs):
        arr = np.array(pairs)  # (batch, 2, dim)
        return np.linalg.norm(arr[:, 0, components] - arr[:, 1, components], axis=-1)

    return Elementwise(Join(cmd, ee), fn)


def cmd_velocity(episode, signal='robot_commands.pose', components=_TRANSLATION, dt_sec=_DT_SEC):
    """Per-frame command translation velocity (m/s). Spikes = tracking glitches."""
    return norm(diff(view(episode.signals[signal], components), dt_sec))


# ---------------------------------------------------------------------------
# Scalar aggregators
# ---------------------------------------------------------------------------


def _signal_values(signal):
    """Extract all values from a Signal as a numpy array."""
    n = len(signal)
    if n == 0:
        return np.array([])
    return np.array(signal._values_at(np.arange(n)))


def agg_max(signal):
    return float(np.max(_signal_values(signal)))


def agg_p95(signal):
    return float(np.percentile(_signal_values(signal), 95))


def agg_mean(signal):
    return float(np.mean(_signal_values(signal)))


def agg_fraction_true(signal):
    """For boolean signals: fraction of True values."""
    return float(np.mean(_signal_values(signal)))
