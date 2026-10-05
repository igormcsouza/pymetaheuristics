from random import Random
from typing import Any, Callable, List, Tuple

Genome = List[Any]
Population = List[Genome]

# selection(population, scores, rng) -> parents; scores are oriented (lower
# is better); bind knobs such as k with functools.partial
SelectionFunction = Callable[[Population, List[float], Random], Population]
# crossover(parent1, parent2, rng) -> (child1, child2)
CrossOverFunction = Callable[
    [Genome, Genome, Random], Tuple[Genome, Genome]]
# mutation(genome, rng) -> genome, the same shape as SA's neighbor
MutationFunction = Callable[[Genome, Random], Genome]
