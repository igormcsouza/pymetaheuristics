# Genetic Algorithm

`genetic_algorithm()` evolves a population of solutions with selection,
crossover and mutation. Use it when good solutions can be combined.

```python
from pymetaheuristics.genetic_algorithm import genetic_algorithm

ga = genetic_algorithm(
    knapsack,
    stop=max_iterations(30),
    rng=1,
    population_size=20,
    mutation=bit_flip_neighbor,
    repair=drop_heaviest,
)
print(ga.best_solution, ga.best_value)
```

Each generation:

1. Breeds `population_size` children. `selection` picks parents and
   `crossover` breeds the first two, until the population is full.
2. Mutates every child. For an infeasible mutant, `mutation` is retried up
   to `max_tries` times, and if every try is infeasible the child is kept
   unmutated.
3. Passes children that are still infeasible through `repair` (if given).
   Any that remain infeasible are replaced by a fresh feasible genome.
4. Evaluates the children. Elitism then puts the best genome so far in
   place of the worst child, so the best is never lost.

| Keyword | Default |
|---|---|
| `population_size` | `10` |
| `selection` | `random_weighted_selection` |
| `crossover` | `single_point_crossover` |
| `mutation` | `inter_mutation` |
| `repair` | `None` |
| `max_tries` | `1000` |

To set an operator's knob, bind it with `functools.partial`, for example
`mutation=partial(inter_mutation, num_swaps=3)`.

!!! warning
    Each generation costs `population_size` evaluations. `stop` is checked
    once per generation, so a `max_evaluations` budget can be exceeded by up
    to one generation.

## Recap

- Pass operators as keywords; defaults work for bit lists.
- Elitism keeps the best genome, and `repair` handles infeasible children.

Next: [Simulated Annealing](simulated-annealing.md).
