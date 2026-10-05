# Extending

The library has no base classes and no registries. Operators, stops and
heuristics are all plain functions, and you pass yours where the built-in
ones would go.

!!! warning
    A custom function must not modify its input. It should return a new
    solution.

## Operator contracts

| Kind | Signature | Passed as |
|---|---|---|
| Neighborhood | `neighbor(solution, rng) -> solution` | `simulated_annealing(neighbor=...)` |
| Cooling | `cooling(temperature) -> temperature` | `simulated_annealing(cooling=...)` |
| Selection | `selection(population, scores, rng, k) -> k parents` | `genetic_algorithm(selection=...)` |
| Crossover | `crossover(parent1, parent2, rng) -> (child1, child2)` | `genetic_algorithm(crossover=...)` |
| Mutation | `mutation(genome, rng) -> genome` | `genetic_algorithm(mutation=...)` |
| Repair | `repair(genome) -> genome` | `genetic_algorithm(repair=...)` |
| Stop | `stop(state) -> bool` | `stop=` of any heuristic |

GA operators are called with `rng` as the last positional argument. Bind
extra knobs with `functools.partial`, for example
`mutation=partial(inter_mutation, num_swaps=3)`. A mutation has the same
shape as an SA `neighbor`, so one function serves both. The GA breeds the
first two parents that selection returns. `scores` holds one value per
genome, already oriented so that lower is better in both directions: a
selection never needs the problem's direction.

Runnable examples, all tested in `tests/examples`:

- [custom_selection.py][selection]: tournament selection.
- [custom_crossover.py][crossover]: order crossover (OX) for TSP.
- [custom_mutation_neighborhood.py][mutation]: one insertion move, used as a
  GA mutation and as an SA neighborhood.
- [custom_constraint_repair.py][repair]: knapsack capacity handled with
  repair and with a penalty.
- [custom_heuristic.py][heuristic]: hill climbing with restarts, written as
  its own loop.

[selection]: https://github.com/igormcsouza/pymetaheuristics/blob/main/examples/custom_selection.py
[crossover]: https://github.com/igormcsouza/pymetaheuristics/blob/main/examples/custom_crossover.py
[mutation]: https://github.com/igormcsouza/pymetaheuristics/blob/main/examples/custom_mutation_neighborhood.py
[repair]: https://github.com/igormcsouza/pymetaheuristics/blob/main/examples/custom_constraint_repair.py
[heuristic]: https://github.com/igormcsouza/pymetaheuristics/blob/main/examples/custom_heuristic.py

## A custom stop

A stop sees each `State` and can keep its own memory. This one ends the run
after `patience` iterations without improvement:

```python
from pymetaheuristics.core import better


def stagnation(patience):
    best, since = None, 0

    def stop(state):
        nonlocal best, since
        if best is None or better(state.best_value, best, state.direction):
            best, since = state.best_value, 0
        else:
            since += 1
        return since >= patience

    return stop
```

## A custom heuristic

A heuristic is any function
`heuristic(problem, *, stop, rng=None, **knobs) -> OptimizationResult`.
The easiest way to write one is with `core.run`. You supply two functions
and `run` takes care of the rest: evaluation counting, timing, stop checks,
tracking the best so far in the problem's direction, `history`, and the
result with `metadata['termination'] == 'stop'` and `metadata['state']`.

```text
init(problem)        -> (carry, solution, value, record)
step(problem, carry) -> (carry, solution, value, record)
```

`carry` is whatever your algorithm threads between iterations.
`solution`/`value` is the iteration's best candidate. `record` is the
iteration's `history` entry, and `None` records the best value so far.
Both functions receive the counted problem.

Here is a random restart search built this way, with an optional
`extras` function that adds to `metadata`:

```python
from pymetaheuristics.core import (
    Problem, any_of, make_rng, max_iterations, reject, run)


def random_search(problem: Problem, *, stop, rng=None, max_tries=1000):
    """Sample fresh feasible solutions and keep the best."""
    rng = make_rng(rng)
    generate = reject(problem.generate, problem.feasible, max_tries)

    def sample(problem, carry):
        solution = generate(rng)
        value = problem.evaluate(solution)
        return carry + 1, solution, value, None

    return run(problem, stop=stop, init=lambda p: sample(p, 0), step=sample,
               extras=lambda samples: {'samples': samples})


problem = Problem(generate=lambda rng: [rng.random()],
                  evaluate=lambda x: (x[0] - 0.5) ** 2)
result = random_search(
    problem, stop=any_of(max_iterations(100), stagnation(10)))
assert result.metadata['termination'] == 'stop'
assert result.metadata['samples'] == result.iterations + 1
assert len(result.history) == result.iterations + 1
```

For a real algorithm to copy, see `simulated_annealing` in
`pymetaheuristics/simulated_annealing/annealing.py`. It is about 30 lines
on top of `run`. [Architecture](architecture.md) explains the contract and
what belongs to the core versus the algorithm. When you need a loop that
`run` can't express, write it yourself. The [custom_heuristic.py][heuristic]
example shows how.
