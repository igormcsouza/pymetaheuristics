from random import Random
from typing import Optional, Union


def make_rng(seed_or_rng: Optional[Union[Random, int]] = None) -> Random:
    """Return a Random: the given one as is, else a new one seeded with it."""
    if isinstance(seed_or_rng, Random):
        return seed_or_rng
    return Random(seed_or_rng)
