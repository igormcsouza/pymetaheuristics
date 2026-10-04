from random import Random
from typing import Optional

from pymetaheuristics.genetic_algorithm.types import Genome
from pymetaheuristics.utils.rng import make_rng


def inter_mutation(
    genome: Genome,
    q: int = 2,
    probability: float = 0.75,
    rng: Optional[Random] = None,
    **kwargs
) -> Genome:
    """At a random chance, change interposition of q genes on the Genome."""
    rng = make_rng(rng)
    for _ in range(q):
        index = rng.randrange(len(genome))

        if rng.random() > probability:
            return genome

        genome[index], genome[index - 1] = genome[index - 1], genome[index]

    return genome
