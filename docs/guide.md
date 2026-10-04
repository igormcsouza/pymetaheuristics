# User guide

All code on this page runs top to bottom, and each block builds on the ones
before it.

## The problem

A `Problem` is the only thing a heuristic knows about your domain. It is a
frozen dataclass of four fields:

| Field | Signature | Meaning |
|---|---|---|
| `generate` | `() -> solution` | a random candidate solution |
| `evaluate` | `(solution) -> float` | the objective value |
| `feasible` | `(solution) -> bool` | whether the solution is allowed (default: always) |
| `direction` | `Direction` | `MINIMIZE` (default) or `MAXIMIZE` |

A solution can be anything your functions understand. The built-in operators
work on lists: bit lists, permutations or lists of floats.

```python
from random import Random

from pymetaheuristics.core import Direction, Problem

VALUES = [60, 100, 120, 80, 30]
WEIGHTS = [10, 20, 30, 25, 5]
CAPACITY = 50
seeded = Random(0)


def weight(packing):
    return sum(w for w, bit in zip(WEIGHTS, packing) if bit)


def value(packing):
    return sum(v for v, bit in zip(VALUES, packing) if bit)


knapsack = Problem(
    generate=lambda: [seeded.randint(0, 1) for _ in VALUES],
    evaluate=value,
    feasible=lambda packing: weight(packing) <= CAPACITY,
    direction=Direction.MAXIMIZE,
)
```

## Objective direction

`direction` controls what "better" means everywhere: selection pressure,
acceptance of moves, the best-so-far, and `target_value` stops. You don't
need to negate a profit to minimize it. Set `Direction.MAXIMIZE` instead.

If your own code needs to compare values, use the same helpers the
heuristics use:

```python
from pymetaheuristics.core import best_of, better, oriented

assert better(220, 180, Direction.MAXIMIZE)
assert best_of([3, 1, 2], Direction.MINIMIZE) == 1
assert oriented(5, Direction.MAXIMIZE) == -5  # lower is always better
```

## Constraints and feasibility

`Problem.feasible` is the contract: **the heuristics only evaluate feasible
solutions.** Infeasible starts are drawn again, infeasible moves are retried
or dropped, and so the best solution returned is always feasible. Pick one
of three strategies, depending on how often `generate` and your operators
produce infeasible solutions.

**Reject.** Draw again until the solution is feasible. This is what the
heuristics already do with `generate`. The `reject` helper wraps any
producer. If no feasible solution turns up within `max_tries`, it raises
`InfeasibleError`.

```python
from pymetaheuristics.core import reject

feasible_packing = reject(knapsack.generate, knapsack.feasible,
                          max_tries=1000)
assert knapsack.feasible(feasible_packing())
```

**Repair.** Turn an infeasible solution into a feasible one. Rejection is
wasteful when most random solutions are infeasible, and repair avoids that.
You can give `genetic_algorithm` a `repair=` function for infeasible
children, or wrap any producer with `repair(produce, fix)`:

```python
from pymetaheuristics.core import repair


def drop_heaviest(packing):
    packing = list(packing)
    while weight(packing) > CAPACITY:
        packed = [i for i, bit in enumerate(packing) if bit]
        packing[max(packed, key=lambda i: WEIGHTS[i])] = 0
    return packing


repaired_generate = repair(knapsack.generate, drop_heaviest,
                           feasible=knapsack.feasible)
assert knapsack.feasible(repaired_generate())
```

**Penalty.** Allow infeasible solutions but score them worse.
`penalty(problem, penalty_fn)` returns a new `Problem` where every solution
is feasible and the objective is worsened by `penalty_fn(solution)` in the
problem's direction. `penalty_fn` must return a value `>= 0` that is zero
for feasible solutions.

```python
from pymetaheuristics.core import penalty

soft = penalty(knapsack, lambda s: 10 * max(0, weight(s) - CAPACITY))
assert soft.evaluate([1, 1, 1, 1, 1]) == 390 - 10 * (90 - 50)
```

With a penalty, the best solution found may be infeasible under the
original constraint. Check it with `knapsack.feasible`.

## Stopping

`stop` is required, and it is a predicate over a `State` snapshot
(`iteration`, `evaluations`, `elapsed`, `best_value`, `direction`). Use the
helpers and combine them with `any_of`:

```python
from pymetaheuristics.core import (
    any_of, max_evaluations, max_iterations, max_time, target_value)

stop = any_of(max_iterations(1000), max_evaluations(5000),
              max_time(2.0), target_value(300))
```

`target_value` compares in the problem's direction, so the target `300`
above means "value at least 300" for this maximization problem. Any
`Callable[[State], bool]` works as a stop. See
[Extending](extending.md#a-custom-stop) for one.

## How heuristics and operators compose

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

## Genetic Algorithm

```python
from pymetaheuristics.genetic_algorithm import genetic_algorithm

ga = genetic_algorithm(
    knapsack,
    stop=max_iterations(30),
    rng=1,
    population_size=20,
    mutation=bit_flip_mutation,
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
| `**operator_kwargs` | forwarded to every operator (with `rng`), e.g. `num_swaps=3` for `inter_mutation`; custom operators must accept `**kwargs` |

Each generation costs `population_size` evaluations. `stop` is checked once
per generation, so a `max_evaluations` budget can be exceeded by up to one
generation.

## Simulated Annealing

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

Tuning: set `initial_temperature` to roughly the size of a typical worsening
move. Choose `cooling` so that the temperature gets close to zero near the
end of your budget.

## Results and history

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

The two x axes count different things. A GA generation costs
`population_size` evaluations, and an SA iteration costs at most one. To
compare the heuristics at equal cost, give both the same
`max_evaluations` stop, as the [experiments](experiments.md) do.

## Reproducibility

Pass `rng=` (an `int` seed or a `random.Random`) to make a run
reproducible. The heuristic passes it to every operator. `problem.generate`
is called **without** an rng, so seed its random source yourself, like the
`seeded = Random(0)` used above.
