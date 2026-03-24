import logging
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Hashable, Sequence
from typing import Any

logger = logging.getLogger(__name__)


class Sampler(ABC):
    """Strategy for selecting which sub-policy to run in a SampledPolicy."""

    @abstractmethod
    def sample(self, keys: Sequence[str], context: dict[str, Any]) -> str:
        """Pick a policy key from available keys given episode context."""

    def count(self, key: str, context: dict[str, Any]):
        """Record a completed episode for the given policy key and context."""
        return None


class UniformSampler(Sampler):
    """Samples policies uniformly at random, optionally with fixed weights per key."""

    def __init__(self, weights: dict[str, float] | None = None):
        self._weights = weights

    def sample(self, keys: Sequence[str], context: dict[str, Any]) -> str:
        w = [self._weights[k] for k in keys] if self._weights else None
        return random.choices(keys, w)[0]


class BalancedSampler(Sampler):
    """Samples policies to balance episode counts across checkpoints.

    For each group (defined by context fields), maintains a counter per policy key.
    Sampling probability for policy i is proportional to:
        max(counts) + balance - count_i
    """

    def __init__(self, *, balance: int = 5, group_fields: Sequence[str] | None = None):
        self._balance = balance
        self.group_fields = tuple(group_fields) if group_fields is not None else None
        self._counts: dict[Hashable, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def _group_key(self, context: dict[str, Any]) -> Hashable:
        if self.group_fields is not None:
            return tuple(context.get(f) for f in self.group_fields)
        return ()

    def sample(self, keys: Sequence[str], context: dict[str, Any]) -> str:
        group = self._group_key(context)
        group_counts = self._counts[group]
        counts = [group_counts[k] for k in keys]
        max_count = max(counts) if counts else 0
        weights = [max_count + self._balance - c for c in counts]
        chosen = random.choices(list(keys), weights)[0]
        lines = [f'BalancedSampler group={group}']
        for k, c, w in zip(keys, counts, weights, strict=True):
            marker = ' ←' if k == chosen else ''
            lines.append(f'  {k}: count={c} weight={w}{marker}')
        logger.info('\n'.join(lines))
        return chosen

    def count(self, key: str, context: dict[str, Any]):
        group = self._group_key(context)
        self._counts[group][key] += 1
