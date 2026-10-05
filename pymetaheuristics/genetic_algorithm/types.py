from random import Random
from typing import Any, Callable, List, Tuple

Genome = List[Any]
Population = List[Genome]

# selection(population, scores, rng, k) -> k parents; scores are oriented
# (lower is better); bind extra knobs with functools.partial
SelectionFunction = Callable[
    [Population, List[float], Random, int], Population]
# crossover(parent1, parent2, rng) -> (child1, child2)
CrossOverFunction = Callable[
    [Genome, Genome, Random], Tuple[Genome, Genome]]
# mutation(genome, rng) -> genome, the same shape as SA's neighbor
MutationFunction = Callable[[Genome, Random], Genome]
