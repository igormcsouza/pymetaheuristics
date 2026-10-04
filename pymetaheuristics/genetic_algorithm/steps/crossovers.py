"""Crossover operators.

Ownership: operators never modify their inputs; the returned Genomes are
always new objects (never aliases of the parents).
"""
from random import Random
from typing import Optional, Tuple

from pymetaheuristics.genetic_algorithm.types import Genome
from pymetaheuristics.genetic_algorithm.exceptions import CrossOverException
from pymetaheuristics.utils.rng import make_rng


def _cut_point(
    parent1: Genome, parent2: Genome, rng: Optional[Random]
) -> Optional[int]:
    """Random cut in ``[1, len - 1]``; None if the genomes are too short.

    Raises CrossOverException if the parents differ in length.
    """
    if len(parent1) != len(parent2):
        raise CrossOverException(
            "Genomes has to have the same length, got %d, %d" % (
                len(parent1), len(parent2)))
    if len(parent1) < 2:
        return None
    return make_rng(rng).randint(1, len(parent1) - 1)


def single_point_crossover(
    parent1: Genome, parent2: Genome, rng: Optional[Random] = None, **kwargs
) -> Tuple[Genome, Genome]:
    """Cut 2 Genomes at a random cut_point and swap their tails."""
    cut_point = _cut_point(parent1, parent2, rng)
    if cut_point is None:
        return parent1[:], parent2[:]

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
    cut_point = _cut_point(parent1, parent2, rng)
    if cut_point is None:
        return parent1[:], parent2[:]

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
