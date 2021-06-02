from pymetaheuristics.genetic_algorithm.exceptions import CrossOverException
from random import choices, randint, randrange, random
from typing import Tuple
from pymetaheuristics.genetic_algorithm.types import (
    FitnessFunction, Genome, Population)


def random_weighted_selection(
    population: Population,
    fitness_function: FitnessFunction,
    k: int = 2
) -> Population:
    return choices(
        population=population,
        weights=[fitness_function(genome) for genome in population],
        k=k
    )


def single_point_crossover(g1: Genome, g2: Genome) -> Tuple[Genome, Genome]:
    if len(g1) == len(g2):
        length = len(g1)
    else:
        raise CrossOverException(
            "Genomes has to have the same length, got %d, %d" % (
                len(g1), len(g2)))

    if length < 2:
        return g1, g2

    p = randint(1, length - 1)

    return g1[0:p] + g2[p:length], g2[0:p] + g1[p:length]


def pmx_single_point(g1: Genome, g2: Genome) -> Tuple[Genome, Genome]:
    """
    PMX implementation for crossover TSP problem.

    See more at
    https://user.ceng.metu.edu.tr/~ucoluk/research/publications/tspnew.pdf ."""
    if len(g1) == len(g2):
        length = len(g1)
    else:
        raise CrossOverException(
            "Genomes has to have the same length, got %d, %d" % (
                len(g1), len(g2)))

    if length < 2:
        return g1, g2

    p = randint(1, length - 1)

    g1child = g1[:]
    for i in range(p):
        ans = g1child.index(g2[i])
        g1child[ans] = g1child[i]
        g1child[i] = g2[i]

    g2child = g2[:]
    for i in range(p):
        ans = g2child.index(g1[i])
        g2child[ans] = g2child[i]
        g2child[i] = g1[i]

    return g1child, g2child


def mutation(genome: Genome, q: int = 2, probability: float = 0.75) -> Genome:
    for _ in range(q):
        index = randrange(len(genome))

        if random() > probability:
            return genome

        genome[index], genome[index - 1] = genome[index - 1], genome[index]

    return genome
