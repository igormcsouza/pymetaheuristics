from random import randint
from typing import Tuple

from pymetaheuristics.genetic_algorithm.types import Genome
from pymetaheuristics.genetic_algorithm.exceptions import CrossOverException


def single_point_crossover(
    g1: Genome, g2: Genome, **kwargs
) -> Tuple[Genome, Genome]:
    """Cut 2 Genomes on index p (randomly choosen) and swap its parts."""
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


def pmx_single_point(
    g1: Genome, g2: Genome, **kwargs
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
