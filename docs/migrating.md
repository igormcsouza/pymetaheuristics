# Migrating from 0.1

The 0.1 names below still work but emit a `DeprecationWarning`, and they
will be removed in 0.3
([#50](https://github.com/igormcsouza/pymetaheuristics/issues/50)).

| 0.1 | Now |
|---|---|
| `GeneticAlgorithm(fitness_function, genome_generator, constraints, direction=...)` | `Problem(generate=..., evaluate=..., feasible=..., direction=...)` |
| `ga.add_constraint(c)` / `constraints=[...]` | one `feasible` predicate: `lambda s: all(c(s) for c in constraints)` |
| `ga.train(epochs, pop_size, ...)` returns `(genome, fitness)` | `genetic_algorithm(problem, stop=max_iterations(epochs), population_size=pop_size, ...)` returns an `OptimizationResult` |
| `ga.history` keyed by timestamp | `result.history`, one dict per generation (see [Results and history](tutorial/results.md)) |
| `verbose=True` | loop over `result.history` after the run |
| `genetic_algorithm.steps.multations` | `genetic_algorithm.steps.mutations` |
| `inter_mutation(genome, q=...)` | `inter_mutation(genome, num_swaps=...)` |
| `utils.distances.euclidian_distance` | `utils.distances.euclidean_distance` (or `math.dist`) |

Before and after:

```text
# 0.1
ga = GeneticAlgorithm(fitness, generate, constraints=[fits])
genome, value = ga.train(epochs=15, pop_size=10, rng=42)
```

```python
from random import Random

from pymetaheuristics.core import Problem, max_iterations
from pymetaheuristics.genetic_algorithm import genetic_algorithm

seeded = Random(0)
problem = Problem(generate=lambda: seeded.sample(range(5), 5),
                  evaluate=lambda s: sum(abs(g - i) for i, g in enumerate(s)))
result = genetic_algorithm(problem, stop=max_iterations(15), rng=42,
                           population_size=10)
genome, value = result.best_solution, result.best_value
```
