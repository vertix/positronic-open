from abc import ABC, abstractmethod
from typing import Any

from positronic.policy.sampler import Sampler, UniformSampler


class Policy(ABC):
    """Abstract base class for all policies."""

    @abstractmethod
    def select_action(self, obs: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        """Computes an action for the given observation.

        **Plain-data contract**
        Policies should accept and return only "plain" data structures:
        - built-in scalars: `str`, `int`, `float`, `bool`, `None`
        - containers: `dict` / `list` / `tuple` recursively composed of supported values
        - numeric numpy values: `numpy.ndarray` and `numpy` scalar types

        Avoid returning arbitrary Python objects (custom classes, sockets, file handles, etc.).
        Keeping inputs/outputs as plain data makes policies easy to compose (wrappers/ensemblers),
        record/replay, and run in different execution contexts.
        """
        pass

    # TODO: Add `dummy_input() -> dict | None` method here. Each vendor Policy subclass
    # (Gr00tPolicy, OpenpiPolicy, DreamZeroPolicy) knows its model's expected input format
    # and can provide a zero-filled dummy for warmup. This removes the need for
    # `Codec.dummy_encoded()` and lets servers warm up without requiring a codec.

    def reset(self, context=None):
        """Resets the policy state."""
        return None

    @property
    def meta(self) -> dict[str, Any]:
        """Returns metadata about the policy configuration."""
        return {}

    def close(self):
        """Closes the policy and releases any resources."""
        return None


class SampledPolicy(Policy):
    """Selects a sub-policy on each reset using a pluggable sampling strategy."""

    def __init__(
        self,
        *policies: Policy,
        sampler: Sampler | None = None,
        weights: list[float] | None = None,
        key_field: str = 'server.checkpoint_path',
    ):
        self._policies = policies
        self._sampler = sampler
        self._weights = weights
        self._key_field = key_field
        self._keys: tuple[str, ...] | None = None
        self._current_key: str | None = None
        self._current_policy = policies[0]

    @property
    def sampler(self) -> Sampler | None:
        if self._sampler is None and self._keys is not None:
            weight_map = dict(zip(self._keys, self._weights, strict=True)) if self._weights else None
            self._sampler = UniformSampler(weight_map)
        return self._sampler

    def _get_keys(self) -> tuple[str, ...]:
        if self._keys is None:
            self._keys = tuple(p.meta.get(self._key_field, str(i)) for i, p in enumerate(self._policies))
        return self._keys

    def select_action(self, obs: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        return self._current_policy.select_action(obs)

    def reset(self, context=None):
        keys = self._get_keys()
        self._current_context = context or {}
        self._current_key = self.sampler.sample(keys, self._current_context)
        self._current_policy = self._policies[keys.index(self._current_key)]
        self._current_policy.reset(context)

    def count_current(self):
        if self._current_key and self._sampler:
            self._sampler.count(self._current_key, self._current_context)

    @property
    def meta(self) -> dict[str, Any]:
        return self._current_policy.meta
