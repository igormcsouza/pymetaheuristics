"""Mutation operators.

Ownership: operators never modify their input; they return a new Genome.
"""
import warnings
from random import Random
from typing import Optional

from pymetaheuristics.genetic_algorithm.types import Genome
from pymetaheuristics.utils.rng import make_rng


def inter_mutation(
    genome: Genome,
    num_swaps: int = 2,
    probability: float = 0.75,
    rng: Optional[Random] = None,
    **kwargs
) -> Genome:
    """At a random chance, swap up to num_swaps gene pairs.

    The input is not modified, a (possibly identical) copy is returned.
    """
    if "q" in kwargs:  # legacy keyword, remove in 0.3 (issue #50)
        warnings.warn("inter_mutation(q=...) is deprecated; use num_swaps.",
                      DeprecationWarning, stacklevel=2)
        num_swaps = kwargs.pop("q")
    rng = make_rng(rng)
    genome = genome[:]
    for _ in range(num_swaps):
        index = rng.randrange(len(genome))

        if rng.random() > probability:
            return genome

        genome[index], genome[index - 1] = genome[index - 1], genome[index]

    return genome
