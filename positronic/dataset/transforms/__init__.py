"""Dataset transformation utilities."""

from positronic.utils.lazy import LazySequence, lazy_sequence

from .dataset import TransformedDataset
from .episode import EpisodeTransform, TransformedEpisode
from .signals import (
    Elementwise,
    IndexOffsets,
    Join,
    TimeOffsets,
    astype,
    concat,
    diff,
    norm,
    pairwise,
    recode_rotation,
    recode_transform,
    view,
)

__all__ = [
    # Signal transforms
    'Elementwise',
    'IndexOffsets',
    'TimeOffsets',
    'Join',
    'LazySequence',
    'lazy_sequence',
    'concat',
    'astype',
    'diff',
    'norm',
    'pairwise',
    'recode_transform',
    'recode_rotation',
    'view',
    # Episode transforms
    'EpisodeTransform',
    'TransformedEpisode',
    # Dataset transforms
    'TransformedDataset',
]
