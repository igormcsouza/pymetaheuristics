# Operators

Heuristics are functions, and operators are the functions you plug into them.

A heuristic is a function `heuristic(problem, *, stop, rng=None, **knobs)`
that returns an `OptimizationResult`. Operators are plain functions that you
pass as keyword arguments, and they never modify their inputs. The library's
operators are:

| Kind | Functions | Module |
|---|---|---|
| Neighborhood (SA move) | `swap_neighbor`, `two_opt_neighbor` (permutations), `bit_flip_neighbor` (bit lists) | `pymetaheuristics.neighborhoods` |
| Cooling (SA) | `geometric_cooling(alpha)`, `linear_cooling(step)` | `pymetaheuristics.simulated_annealing` |
| Selection (GA) | `random_weighted_selection` | `pymetaheuristics.genetic_algorithm.steps.selections` |
| Crossover (GA) | `single_point_crossover`, `pmx_single_point` (permutations) | `pymetaheuristics.genetic_algorithm.steps.crossovers` |
| Mutation (GA) | `inter_mutation` (swaps adjacent genes) | `pymetaheuristics.genetic_algorithm.steps.mutations` |

`pymetaheuristics.simulated_annealing` re-exports the three neighborhoods.

A neighborhood is `neighbor(solution, rng)`, and a GA mutation has the
same shape, `mutation(genome, rng)`, so a neighborhood plugs straight in as `mutation=`, and knobs are bound
with `functools.partial`:

```python
from functools import partial
from random import Random

from pymetaheuristics.genetic_algorithm.steps.mutations import inter_mutation
from pymetaheuristics.neighborhoods import bit_flip_neighbor

mutation = partial(inter_mutation, num_swaps=3)
print(bit_flip_neighbor([0, 0, 0], Random(0)), mutation([1, 2, 3], Random(0)))
```

The other GA operators are `selection(population, scores, rng, k)` (returns
`k` parents, the GA asks for 2; scores are oriented, lower is better) and `crossover(parent1, parent2, rng)`.

## Recap

- Operators are plain functions that never modify their inputs.
- A neighborhood is a valid GA mutation as is.

Next: [Genetic Algorithm](genetic-algorithm.md) or
[Simulated Annealing](simulated-annealing.md).
