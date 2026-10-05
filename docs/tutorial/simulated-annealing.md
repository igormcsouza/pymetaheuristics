# Simulated Annealing

`simulated_annealing()` walks from one solution to a neighbor, sometimes
accepting a worse one to escape local optima. It is cheap per iteration.

```python
from pymetaheuristics.simulated_annealing import (
    geometric_cooling, simulated_annealing)

sa = simulated_annealing(
    knapsack,
    stop=max_iterations(2000),
    rng=1,
    neighbor=bit_flip_neighbor,
    initial_temperature=50.0,
    cooling=geometric_cooling(0.995),
)
print(sa.best_solution, sa.best_value)
```

A worse move of size `d` (measured in the problem's direction) is accepted
with probability `exp(-d / T)`. After each iteration, the temperature
becomes `cooling(T)`. An infeasible neighbor is redrawn up to
`max_neighbor_tries` times (default 100). If no feasible neighbor turns up,
the iteration keeps the current solution. Each iteration costs at most one
evaluation.

!!! tip "Tuning"
    Set `initial_temperature` to roughly the size of a typical worsening
    move. Choose `cooling` so that the temperature gets close to zero near
    the end of your budget.

## Recap

- You supply a `neighbor` move and a `cooling` schedule.
- Each iteration costs at most one evaluation.

Next: [read the results](results.md).
