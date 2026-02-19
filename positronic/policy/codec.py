"""Composable codec for encoding observations and decoding actions.

A Codec pairs an observation encoder (for training and inference) with an action decoder.
Codecs compose via ``|``: ``observation_codec | action_codec | timing`` produces a single
codec that encodes left-to-right and decodes right-to-left.
"""

from abc import ABC, abstractmethod
from typing import Any, final

from positronic.dataset.transforms.episode import Derive, EpisodeTransform, Group
from positronic.policy.base import Policy
from positronic.utils import merge_dicts


def lerobot_state(dim: int, names: list[str] | None = None) -> dict[str, Any]:
    """LeRobot feature descriptor for a state vector."""
    f: dict[str, Any] = {'shape': (dim,), 'dtype': 'float32'}
    if names:
        f['names'] = names
    return f


def lerobot_image(width: int, height: int) -> dict[str, Any]:
    """LeRobot feature descriptor for an RGB image."""
    return {'shape': (height, width, 3), 'names': ['height', 'width', 'channel'], 'dtype': 'video'}


def lerobot_action(dim: int) -> dict[str, Any]:
    """LeRobot feature descriptor for an action vector."""
    return {'shape': (dim,), 'names': ['actions'], 'dtype': 'float32'}


class Codec(ABC):
    """Base class for observation/action codecs.

    Subclasses implement ``encode`` (observation encoding or pass-through for action codecs)
    and optionally ``_decode_single`` (action decoding). The ``training_encoder`` property
    returns an ``EpisodeTransform`` used by the training pipeline to derive dataset columns.
    """

    @abstractmethod
    def encode(self, data: dict) -> dict: ...

    def decode(self, data, *, context=None):
        if isinstance(data, list):
            return [self.decode(d, context=context) for d in data]
        return self._decode_single(data, context)

    def _decode_single(self, data: dict, context: dict | None) -> dict:
        return data

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Derive()

    @property
    def meta(self) -> dict:
        return {}

    def dummy_input(self) -> dict | None:
        return None

    @final
    def wrap(self, policy: Policy) -> Policy:
        return _WrappedPolicy(policy, self)

    @final
    def __or__(self, other: 'Codec') -> 'Codec':
        return _ComposedCodec(self, other)


class _ComposedCodec(Codec):
    """Two codecs composed via ``|``. Encodes left-to-right, decodes right-to-left."""

    def __init__(self, left: Codec, right: Codec):
        self._left = left
        self._right = right

    def encode(self, data):
        return self._right.encode(self._left.encode(data))

    def decode(self, data, *, context=None):
        return self._left.decode(self._right.decode(data, context=context), context=context)

    @property
    def training_encoder(self):
        return Group(self._left.training_encoder, self._right.training_encoder)

    @property
    def meta(self):
        result: dict[str, Any] = {}
        merge_dicts(result, self._left.meta)
        merge_dicts(result, self._right.meta)
        return result

    def dummy_input(self):
        return self._left.dummy_input() or self._right.dummy_input()


class ActionTiming(Codec):
    """Attaches timings to decoded actions and truncates action sequences to a specified horizon.

    At inference time, truncates action chunks to ``horizon`` seconds and stamps each action
    with a ``timestamp`` field. At training time, surfaces ``action_fps`` (and optionally
    ``action_horizon_sec``) as transform metadata so the training pipeline can read it.
    """

    def __init__(self, *, fps: float, horizon: float | None = None):
        self._fps = fps
        self._horizon = horizon

    def encode(self, data):
        return data

    def decode(self, data, *, context=None):
        if isinstance(data, list):
            if self._horizon is not None:
                data = data[: round(self._horizon * self._fps)]
            dt = 1.0 / self._fps
            for i, d in enumerate(data):
                d['timestamp'] = i * dt
            return data
        data['timestamp'] = 0.0
        return data

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Derive(meta=self.meta)

    @property
    def meta(self):
        result = {'action_fps': self._fps}
        if self._horizon is not None:
            result['action_horizon_sec'] = self._horizon
        return result


class _WrappedPolicy(Policy):
    """Policy wrapped with a codec: encodes observations, decodes actions."""

    def __init__(self, policy: Policy, codec: Codec):
        self._policy = policy
        self._codec = codec

    def select_action(self, obs):
        encoded = self._codec.encode(obs)
        action = self._policy.select_action(encoded)
        return self._codec.decode(action, context=obs)

    def reset(self):
        self._policy.reset()

    @property
    def meta(self):
        return self._policy.meta | self._codec.meta

    def close(self):
        self._policy.close()
