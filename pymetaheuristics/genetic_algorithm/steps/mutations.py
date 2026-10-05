"""Mutation operators.

Ownership: operators never modify their input; they return a new Genome.
"""
from random import Random

from pymetaheuristics.genetic_algorithm.types import Genome


def inter_mutation(
    genome: Genome, rng: Random, num_swaps: int = 2, probability: float = 0.75
) -> Genome:
    """At a random chance, swap up to num_swaps gene pairs.

    The input is not modified, a (possibly identical) copy is returned.
    """
    genome = genome[:]
    for _ in range(num_swaps):
        index = rng.randrange(len(genome))

        if rng.random() > probability:
            return genome

        genome[index], genome[index - 1] = genome[index - 1], genome[index]

    return genome
