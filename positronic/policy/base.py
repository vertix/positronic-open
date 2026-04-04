from abc import ABC, abstractmethod
from typing import Any

from positronic.policy.sampler import Sampler, UniformSampler


class Session(ABC):
    """Per-episode inference session. Created by ``Policy.new_session()``.

    Sessions hold per-episode state (trajectory buffers, latency tracking, etc.)
    and are the primary interface for running inference. Call the session like
    a function to get actions::

        session = policy.new_session(context)
        trajectory = session(obs)

    **Plain-data contract**: sessions accept and return only plain data
    (dicts, lists, numpy arrays, scalars). No tensors or custom objects.
    """

    @abstractmethod
    def __call__(self, obs: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        """Predict actions for the given observation."""

    @property
    def meta(self) -> dict[str, Any]:
        """Session metadata (may include policy meta + per-session info)."""
        return {}

    def on_episode_complete(self):  # noqa: B027
        """Called when the episode using this session is finalized."""

    def close(self):  # noqa: B027
        """End this session and release per-episode resources."""


class Policy(ABC):
    """Factory for inference sessions.

    A Policy holds shared resources (model weights, connections) and creates
    per-episode ``Session`` instances. One Policy can serve multiple robots
    by creating independent sessions.
    """

    @abstractmethod
    def new_session(self, context: dict[str, Any] | None = None) -> Session:
        """Create a new inference session for an episode.

        Args:
            context: Episode context (task description, eval metadata, etc.).
        """

    @property
    def meta(self) -> dict[str, Any]:
        """Static metadata about this policy/model."""
        return {}

    def close(self):  # noqa: B027
        """Release shared resources (model weights, connections, etc.)."""


class _SampledSession(Session):
    """Session created by SampledPolicy — wraps an inner session + tracks sampling."""

    def __init__(self, inner: Session, key: str, sampler: Sampler, context: dict):
        self._inner = inner
        self._key = key
        self._sampler = sampler
        self._context = context

    def __call__(self, obs):
        return self._inner(obs)

    @property
    def meta(self):
        return self._inner.meta

    def on_episode_complete(self):
        self._sampler.count(self._key, self._context)
        self._inner.on_episode_complete()

    def close(self):
        self._inner.close()


class SampledPolicy(Policy):
    """Selects a sub-policy on each new_session using a pluggable sampling strategy."""

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

    def new_session(self, context=None):
        keys = self._get_keys()
        ctx = context or {}
        key = self.sampler.sample(keys, ctx)
        sub_policy = self._policies[keys.index(key)]
        inner_session = sub_policy.new_session(context)
        return _SampledSession(inner_session, key, self.sampler, ctx)

    @property
    def meta(self):
        return {}
