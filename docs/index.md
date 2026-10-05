# pymetaheuristics

Metaheuristics for optimization problems in plain Python, with no
dependencies. You describe the problem: how to make a solution, how to score
it, which solutions are allowed, and whether to minimize or maximize. Then
you hand it to a heuristic, which returns the best solution it found.

Metaheuristics are a good fit when the search space is too large to
enumerate and no exact solver suits the problem: combinatorial problems
(routing, packing, scheduling, assignment) or black-box objectives. They
give good solutions quickly, but they don't prove those solutions are
optimal.

The library ships with:

- **Genetic Algorithm**: `genetic_algorithm()`, a population-based search
  with pluggable selection, crossover and mutation.
- **Simulated Annealing**: `simulated_annealing()`, a single-trajectory
  search with pluggable neighborhoods and cooling schedules.
- A [benchmark suite](benchmarks.md) of Knapsack, TSP and continuous
  problems with known optima.

Every heuristic is a plain function with the same shape, so you can swap
algorithms without touching the problem. This documentation describes the
API as of 0.2.

## Install

Requires Python 3.12+.

=== "pip"

    ```sh
    pip install pymetaheuristics
    ```

=== "uv"

    ```sh
    uv add pymetaheuristics
    ```

## Quickstart

Find a short closed tour through five points:

```python
import math
from pymetaheuristics.core import Problem, max_iterations
from pymetaheuristics.simulated_annealing import simulated_annealing

cities = [(0, 0), (0, 2), (3, 2), (3, 0), (1, 1)]


def tour_length(tour):
    return sum(math.dist(cities[tour[i - 1]], cities[tour[i]])
               for i in range(len(tour)))


problem = Problem(
    generate=lambda rng: rng.sample(range(len(cities)), len(cities)),
    evaluate=tour_length,
)  # direction defaults to Direction.MINIMIZE

result = simulated_annealing(problem, stop=max_iterations(500), rng=0)
print(result.best_solution, round(result.best_value, 3))
```

Switching heuristics changes only the call:

=== "Simulated Annealing"

    ```python
    result = simulated_annealing(problem, stop=max_iterations(500), rng=0)
    ```

=== "Genetic Algorithm"

    ```python
    from pymetaheuristics.genetic_algorithm import genetic_algorithm
    from pymetaheuristics.genetic_algorithm.steps.crossovers import (
        pmx_single_point)

    result = genetic_algorithm(problem, stop=max_iterations(50), rng=0,
                               crossover=pmx_single_point)
    ```

## What can I do?

| I want to... | Go to |
|---|---|
| describe my problem, minimizing or maximizing | [Problem and direction](tutorial/problem.md) |
| forbid invalid solutions (reject, repair, penalty) | [Constraints](tutorial/constraints.md) |
| stop by iterations, evaluations, time or target value | [Stopping](tutorial/stopping.md) |
| use a built-in neighborhood, selection, crossover or mutation | [Operators](tutorial/operators.md) |
| search with a population | [Genetic Algorithm](tutorial/genetic-algorithm.md) |
| search with one moving solution | [Simulated Annealing](tutorial/simulated-annealing.md) |
| read the best solution and plot convergence | [Results and history](tutorial/results.md) |
| get the same run twice | [Reproducibility](tutorial/reproducibility.md) |
| see complete programs | [Worked examples](examples.md) |
| write my own operator, stop or heuristic | [Extending](extending.md) |
| test against problems with known optima | [Benchmark suite](benchmarks.md) |
| look up a function | [Reference](reference/index.md) |

## Where next

- [Tutorial](tutorial/index.md): problems, directions, constraints, both
  heuristics, results and history.
- [Worked examples](examples.md): Knapsack and TSP from start to finish.
- [Extending](extending.md): write your own operator or heuristic.
- [Reference](reference/index.md): the API, from the docstrings.
- [Architecture](architecture.md): how the core is put together.
- [Release notes](release-notes.md): what changed between versions.
