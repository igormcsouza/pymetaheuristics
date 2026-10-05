# Artificial Bee Colony

`artificial_bee_colony()` keeps a colony of solutions and improves them with
three kinds of bees. Use it when you want a population method with a single
knob for how long to persist on a solution, and no crossover to design.

```python
from pymetaheuristics.artificial_bee_colony import artificial_bee_colony

abc = artificial_bee_colony(
    knapsack,
    stop=max_iterations(50),
    rng=1,
    neighbor=bit_flip_neighbor,
    colony_size=10,
    limit=20,
)
print(abc.best_solution, abc.best_value)
```

Each iteration:

1. **Employed bees:** every solution tries one neighbor and keeps it only if
   it is better.
2. **Onlooker bees:** `colony_size` more tries, each on a solution picked by a
   tournament between two random ones, so good solutions get more tries.
3. **Scouts:** a solution that failed more than `limit` times in a row is
   replaced by a fresh one from `problem.generate`.

The `neighbor` is the same `neighbor(solution, rng)` function that simulated
annealing uses, so any neighborhood works. An infeasible neighbor is redrawn
up to `max_neighbor_tries` times (default 100), and counts as a failed try if
none is feasible.

One iteration costs about `2 * colony_size` evaluations (more when scouts
fire), and `stop` is checked once per iteration, so `max_evaluations` may be
overshot by up to one iteration.

!!! tip "Tuning"
    `colony_size` and `limit` have generic defaults (20 and 50). A smaller
    colony gives each solution more tries for a fixed budget, and a smaller
    `limit` restarts stuck solutions sooner. For continuous problems the step
    of the neighbor (for example `gaussian_neighbor(sigma=...)`) matters as
    much.

For the equations, the differences from the original algorithm and measured
results, see [Artificial Bee Colony in depth](../algorithms/artificial-bee-colony.md).

## Recap

- You supply a `neighbor` move; `colony_size` and `limit` are the knobs.
- Each iteration costs about `2 * colony_size` evaluations.

Next: [read the results](results.md).
