"""Crossover operators.

Ownership: operators never modify their inputs; the returned Genomes are
always new objects (never aliases of the parents).
"""
from random import Random
from typing import Optional, Tuple

from pymetaheuristics.genetic_algorithm.types import Genome
from pymetaheuristics.genetic_algorithm.exceptions import CrossOverException
from pymetaheuristics.utils.rng import make_rng


def single_point_crossover(
    parent1: Genome, parent2: Genome, rng: Optional[Random] = None, **kwargs
) -> Tuple[Genome, Genome]:
    """Cut 2 Genomes at a random cut_point and swap their tails."""
    if len(parent1) == len(parent2):
        length = len(parent1)
    else:
        raise CrossOverException(
            "Genomes has to have the same length, got %d, %d" % (
                len(parent1), len(parent2)))

    if length < 2:
        return parent1[:], parent2[:]

    cut_point = make_rng(rng).randint(1, length - 1)

    return (parent1[:cut_point] + parent2[cut_point:],
            parent2[:cut_point] + parent1[cut_point:])


def pmx_single_point(
    parent1: Genome, parent2: Genome, rng: Optional[Random] = None, **kwargs
) -> Tuple[Genome, Genome]:
    """
    PMX is a crossover function which consider a Genome as a sequence of
    nom-repetitive genes through the Genome. So before swapping, checks if
    repetition is going to occur, and swap the pretitive gene with its partner
    on the other Genome and them swap with other gene on the same Genome.

    See more at
    https://user.ceng.metu.edu.tr/~ucoluk/research/publications/tspnew.pdf .

    This implementation suites very well the TSP problem.
    """
    if len(parent1) == len(parent2):
        length = len(parent1)
    else:
        raise CrossOverException(
            "Genomes has to have the same length, got %d, %d" % (
                len(parent1), len(parent2)))

    if length < 2:
        return parent1[:], parent2[:]

    cut_point = make_rng(rng).randint(1, length - 1)

    child1 = parent1[:]
    for i in range(cut_point):
        partner_index = child1.index(parent2[i])
        child1[partner_index] = child1[i]
        child1[i] = parent2[i]

    child2 = parent2[:]
    for i in range(cut_point):
        partner_index = child2.index(parent1[i])
        child2[partner_index] = child2[i]
        child2[i] = parent1[i]

    return child1, child2
