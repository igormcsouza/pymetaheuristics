"""Mutation operators.

Ownership: operators never modify their input; they return a new Genome.
"""
from random import randrange, random

from pymetaheuristics.genetic_algorithm.types import Genome


def inter_mutation(
    genome: Genome,
    q: int = 2,
    probability: float = 0.75,
    **kwargs
) -> Genome:
    """At a random chance, change interposition of q genes on the Genome.

    The input is not modified, a (possibly identical) copy is returned.
    """
    genome = genome[:]
    for _ in range(q):
        index = randrange(len(genome))

        if random() > probability:
            return genome

        genome[index], genome[index - 1] = genome[index - 1], genome[index]

    return genome
