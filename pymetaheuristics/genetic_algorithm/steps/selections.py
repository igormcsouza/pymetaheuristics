"""Selection operators.

Ownership: operators never modify the population; the returned Genomes are
copies, not aliases of the population members.
"""
from random import Random
from typing import Optional

from pymetaheuristics.core.direction import oriented
from pymetaheuristics.core.problem import Direction
from pymetaheuristics.genetic_algorithm.types import (
    FitnessFunction, Population)
from pymetaheuristics.utils.rng import make_rng


def random_weighted_selection(
    population: Population,
    fitness_function: FitnessFunction,
    k: int = 2,
    rng: Optional[Random] = None,
    direction: Direction = Direction.MINIMIZE,
    **kwargs
) -> Population:
    """Selects randomly k genomes on a population. This approach considers the
    fitness of each Genome as weights so the most fitted is very likely to be
    choosen, but, still gives room for a little of jumps.
    """
    # oriented: lower is better for either direction
    fitness = [
        oriented(fitness_function(genome), direction) for genome in population]
    # lower => higher weight; shift keeps weights > 0
    selected = make_rng(rng).choices(
        population=population,
        weights=[max(fitness) - f + 1 for f in fitness],
        k=k
    )
    return [genome[:] for genome in selected]
