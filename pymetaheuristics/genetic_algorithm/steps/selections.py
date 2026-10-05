"""Selection operators.

Ownership: operators never modify the population; the returned Genomes are
copies, not aliases of the population members.
"""
from random import Random
from typing import List

from pymetaheuristics.genetic_algorithm.types import Population

_SHIFT = 0.1  # weight of the worst genome, as a fraction of the range


def random_weighted_selection(
    population: Population, scores: List[float], rng: Random, k: int = 2
) -> Population:
    """Selects randomly k genomes on a population. This approach considers the
    fitness of each Genome as weights so the most fitted is very likely to be
    choosen, but, still gives room for a little of jumps.
    """
    # lower => higher weight. Shift by a fraction of the value range (not a
    # constant) so selection pressure does not depend on the objective's
    # scale; the shift keeps the worst genome selectable (weight > 0).
    worst, spread = max(scores), max(scores) - min(scores)
    weights = [worst - value + _SHIFT * spread for value in scores]
    selected = rng.choices(
        population=population,
        weights=weights if spread > 0 else None,  # all equal: uniform
        k=k
    )
    return [genome[:] for genome in selected]
