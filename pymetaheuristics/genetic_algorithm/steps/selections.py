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
    fitness = [fitness_function(genome) for genome in population]
    # minimization: lower fitness => higher weight; shift keeps weights > 0
    return choices(
        population=population,
        weights=[max(fitness) - f + 1 for f in fitness],
        k=k
    )
