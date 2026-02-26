"""Per-frame quality signals and scalar aggregators for dataset debugging.

Quality signals are lazy Signal views composed from Elementwise, TimeOffsets,
and Join — nothing expensive happens until values are accessed. Scalar
aggregators extract all values from a signal and compute a summary statistic.
"""

import numpy as np

from .signals import Elementwise, Join, TimeOffsets

_TRANSLATION = slice(0, 3)


def _dt(fps):
    """Time delta in nanoseconds for a given analysis fps."""
    return int(1e9 / fps)


def idle_mask(episode, signal='robot_state.q', velocity_threshold=0.015, fps=15):
    """Per-frame bool: True where joint velocity < threshold (rad/s).

    ``fps`` controls the analysis timescale (time window for velocity
    computation).  A threshold of 0.015 rad/s at 15 fps means the same
    physical velocity regardless of the source signal's native rate.
    """
    q = episode.signals[signal]
    dt = _dt(fps)

    def fn(pairs):
        arr = np.array(pairs)  # (batch, 2, dim)
        velocity = np.linalg.norm(arr[:, 1] - arr[:, 0], axis=-1) / (2 / fps)
        return velocity < velocity_threshold

    return Elementwise(TimeOffsets(q, -dt, dt), fn)


def jerk(episode, signal='robot_state.q', fps=15):
    """Per-frame joint acceleration magnitude (rad/s^2)."""
    q = episode.signals[signal]
    dt = _dt(fps)
    fps2 = fps * fps

    def fn(triples):
        arr = np.array(triples)  # (batch, 3, dim)
        accel = (arr[:, 2] - 2 * arr[:, 1] + arr[:, 0]) * fps2
        return np.linalg.norm(accel, axis=-1)

    return Elementwise(TimeOffsets(q, -dt, 0, dt), fn)


def cmd_lag(episode, cmd_signal='robot_commands.pose', state_signal='robot_state.ee_pose', components=_TRANSLATION):
    """Per-frame distance between commanded and actual pose (meters).

    No fps needed — this is an instantaneous metric (not a derivative).
    """
    cmd = episode.signals[cmd_signal]
    ee = episode.signals[state_signal]

    def fn(pairs):
        arr = np.array(pairs)  # (batch, 2, dim)
        return np.linalg.norm(arr[:, 0, components] - arr[:, 1, components], axis=-1)

    return Elementwise(Join(cmd, ee), fn)


def cmd_velocity(episode, signal='robot_commands.pose', components=_TRANSLATION, fps=15):
    """Per-frame command translation velocity (m/s). Spikes = tracking glitches."""
    cmd = episode.signals[signal]
    dt = _dt(fps)

    def fn(pairs):
        arr = np.array(pairs)  # (batch, 2, dim)
        return np.linalg.norm(arr[:, 1, components] - arr[:, 0, components], axis=-1) / (2 / fps)

    return Elementwise(TimeOffsets(cmd, -dt, dt), fn)


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
