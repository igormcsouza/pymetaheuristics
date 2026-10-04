"""Selection operators.

Ownership: operators never modify the population; the returned Genomes are
copies, not aliases of the population members.
"""
from random import choices

from pymetaheuristics.genetic_algorithm.types import (
    FitnessFunction, Population)


def random_weighted_selection(
    population: Population,
    fitness_function: FitnessFunction,
    k: int = 2,
    **kwargs
) -> Population:
    """Selects randomly k genomes on a population. This approach considers the
    fitness of each Genome as weights so the most fitted is very likely to be
    choosen, but, still gives room for a little of jumps.
    """
    selected = choices(
        population=population,
        weights=[-fitness_function(genome) for genome in population],
        k=k
    )
    return [genome[:] for genome in selected]
