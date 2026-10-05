# Results and history

Every heuristic returns the same `OptimizationResult`, so you read and
compare runs the same way.

Every heuristic returns an `OptimizationResult`:

| Field | Meaning |
|---|---|
| `best_solution` | best feasible solution found |
| `best_value` | its objective value |
| `history` | one record per iteration; the initial state comes first, so `len(history) == iterations + 1` |
| `iterations` | completed iterations (GA: generations) |
| `elapsed` | wall time in seconds |
| `metadata` | `evaluations` (objective calls), `termination` (`'stop'`), `state` (the final `State`, the one that satisfied `stop`), plus algorithm extras |

`metadata['state']` tells you which budget ended the run:

```python
state = sa.metadata['state']
print(sa.metadata['termination'], state.iteration, state.evaluations)
print('final temperature:', sa.metadata['final_temperature'])  # SA extra
```

The two heuristics record different things in `history`:

- **Simulated Annealing**: a float per iteration, the best value so far.
- **Genetic Algorithm**: a dict per generation:

```text
{'best': float,        # this generation's best value
 'mean': float,        # mean value of the population
 'worst': float,       # worst value of the population
 'solution': genome,   # this generation's best genome
 'best_so_far': float} # best value up to and including this generation
```

`best_so_far` gives both heuristics a common convergence measure:

```python
def convergence(result):
    return [r['best_so_far'] if isinstance(r, dict) else r
            for r in result.history]


assert convergence(ga)[-1] == ga.best_value
assert convergence(sa)[-1] == sa.best_value
```

To plot it (`pip install matplotlib`):

```python
import matplotlib.pyplot as plt

plt.plot(convergence(sa), label='simulated annealing (per iteration)')
plt.plot(convergence(ga), label='genetic algorithm (per generation)')
plt.xlabel('iteration')
plt.ylabel('best value so far')
plt.legend()
plt.show()
```

!!! warning
    The two x axes count different things. A GA generation costs
    `population_size` evaluations, and an SA iteration costs at most one. To
    compare the heuristics at equal cost, give both the same
    `max_evaluations` stop, as the [experiments](../experiments.md) do.

## Recap

- `best_solution`, `best_value`, `history`, `iterations`, `elapsed`, `metadata`.
- `best_so_far` gives both heuristics a common convergence curve.

Next: [reproducibility](reproducibility.md).
