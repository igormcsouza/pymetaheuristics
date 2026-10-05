# Worked examples

Two complete programs. Each one models a problem, solves it with both
heuristics, and checks the answer against the known optimum. The test suite
runs these blocks, so the code on this page stays working. For smaller
programs that extend the library, see [Extending](extending.md).

## 0/1 Knapsack (maximize, constrained)

Pick items to maximize their total value without going over the capacity.
A solution is a list of bits, with 1 meaning the item is packed.

```python
from pymetaheuristics.core import Direction, Problem, max_iterations
from pymetaheuristics.genetic_algorithm import genetic_algorithm
from pymetaheuristics.neighborhoods import bit_flip_neighbor
from pymetaheuristics.simulated_annealing import (
    geometric_cooling, simulated_annealing)

VALUES = [55, 10, 47, 5, 4, 50, 8, 61, 85, 87]
WEIGHTS = [95, 4, 60, 32, 23, 72, 80, 62, 65, 46]
CAPACITY = 269


def total(data, packing):
    return sum(d for d, bit in zip(data, packing) if bit)


def random_packing(rng):
    """Add items in random order while they fit: always feasible."""
    packing = [0] * len(VALUES)
    for i in rng.sample(range(len(VALUES)), len(VALUES)):
        packing[i] = 1
        if total(WEIGHTS, packing) > CAPACITY:
            packing[i] = 0
    return packing


problem = Problem(
    generate=random_packing,
    evaluate=lambda packing: total(VALUES, packing),
    feasible=lambda packing: total(WEIGHTS, packing) <= CAPACITY,
    direction=Direction.MAXIMIZE,
)


ga = genetic_algorithm(problem, stop=max_iterations(50), rng=0,
                       population_size=20, mutation=bit_flip_neighbor)
sa = simulated_annealing(problem, stop=max_iterations(2000), rng=0,
                         neighbor=bit_flip_neighbor,
                         initial_temperature=50.0,
                         cooling=geometric_cooling(0.995))

for name, result in [('GA', ga), ('SA', sa)]:
    print(name, result.best_solution, result.best_value,
          result.metadata['evaluations'], 'evaluations')
    assert problem.feasible(result.best_solution)
    assert result.best_value == 295  # optimum, by brute force over 2^10
```

!!! note
    - `generate` only produces feasible packings, so no start is ever rejected.
      Bit flips and crossover can still overfill the knapsack. The GA retries
      such mutants and replaces children that stay infeasible. SA redraws
      infeasible neighbors.
    - This instance is also available as `get('knapsack-10')` in
      `pymetaheuristics.benchmarks`, and `knapsack(values, weights, capacity)`
      builds other instances. See [Benchmarks](benchmarks.md).
    - [Tutorial: constraints](tutorial/constraints.md) shows the
      same problem solved with repair and with a penalty.

## Travelling Salesman (minimize, permutations)

Visit every city exactly once and return to the start, using the shortest
closed tour. A solution is a permutation of city indices.

```python
import math
from pymetaheuristics.core import (
    Problem, any_of, max_evaluations, target_value)
from pymetaheuristics.genetic_algorithm import genetic_algorithm
from pymetaheuristics.genetic_algorithm.steps.crossovers import (
    pmx_single_point)
from pymetaheuristics.neighborhoods import two_opt_neighbor
from pymetaheuristics.simulated_annealing import (
    geometric_cooling, simulated_annealing)

CITIES = [(x, y) for x in range(3) for y in range(3)]  # a 3x3 grid
OPTIMUM = 8 + math.sqrt(2)


def tour_length(tour):
    return sum(math.dist(CITIES[tour[i - 1]], CITIES[tour[i]])
               for i in range(len(tour)))


problem = Problem(
    generate=lambda rng: rng.sample(range(len(CITIES)), len(CITIES)),
    evaluate=tour_length,
    feasible=lambda tour: sorted(tour) == list(range(len(CITIES))),
)  # Direction.MINIMIZE is the default


# Stop at the optimum, or after 3000 evaluations at the latest.
stop = any_of(target_value(OPTIMUM + 1e-9), max_evaluations(3000))

ga = genetic_algorithm(problem, stop=stop, rng=0, population_size=20,
                       crossover=pmx_single_point, mutation=two_opt_neighbor)
sa = simulated_annealing(problem, stop=stop, rng=0,
                         neighbor=two_opt_neighbor, initial_temperature=1.0,
                         cooling=geometric_cooling(0.998))

for name, result in [('GA', ga), ('SA', sa)]:
    print(name, result.best_solution, round(result.best_value, 3),
          result.metadata['evaluations'], 'evaluations')
    assert math.isclose(result.best_value, OPTIMUM)
```

!!! note
    - `pmx_single_point` and `two_opt_neighbor` always return permutations, so
      `feasible` only acts as a safety net. With a crossover that does not
      preserve permutations, such as `single_point_crossover`, infeasible
      children would be replaced every generation and the GA would degrade into
      random search.
    - `target_value` ends the run as soon as the optimum is reached.
      `metadata['evaluations']` shows how many evaluations each heuristic
      needed.
    - `tsp(cities)` in `pymetaheuristics.benchmarks` builds the same `Problem`
      from a list of points.

To compare the two runs visually, plot `best_so_far` as shown in
[Results and history](tutorial/results.md).
