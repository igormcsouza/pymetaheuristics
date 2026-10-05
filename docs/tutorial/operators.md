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

A neighborhood is `neighbor(solution, rng)`, and a GA mutation is
`mutation(genome, rng=..., **kwargs)`. To reuse a neighborhood as a
mutation, adapt it in one line:

```python
from pymetaheuristics.neighborhoods import bit_flip_neighbor


def bit_flip_mutation(genome, rng, **kwargs):
    return bit_flip_neighbor(genome, rng)
```

## Recap

- Operators are plain functions that never modify their inputs.
- Reuse a neighborhood as a GA mutation with a one-line adapter.

Next: [Genetic Algorithm](genetic-algorithm.md) or
[Simulated Annealing](simulated-annealing.md).
