# Problem and direction

You describe your problem once as a `Problem`. Every heuristic in the
library can then solve it. This page defines one, and shows how to choose
whether lower or higher values are better.

!!! info
    All code in the tutorial runs top to bottom, and each block builds on the
    ones before it. Copy the blocks in order into one file to follow along.

## The problem

A `Problem` is the only thing a heuristic knows about your domain. It is a
frozen dataclass of four fields:

| Field | Signature | Meaning |
|---|---|---|
| `generate` | `(rng) -> solution` | a random candidate solution |
| `evaluate` | `(solution) -> float` | the objective value |
| `feasible` | `(solution) -> bool` | whether the solution is allowed (default: always) |
| `direction` | `Direction` | `MINIMIZE` (default) or `MAXIMIZE` |

!!! tip
    A solution can be anything your functions understand. The built-in
    operators work on lists: bit lists, permutations or lists of floats.

```python
from pymetaheuristics.core import Direction, Problem

VALUES = [60, 100, 120, 80, 30]
WEIGHTS = [10, 20, 30, 25, 5]
CAPACITY = 50


def weight(packing):
    return sum(w for w, bit in zip(WEIGHTS, packing) if bit)


def value(packing):
    return sum(v for v, bit in zip(VALUES, packing) if bit)


knapsack = Problem(
    generate=lambda rng: [rng.randint(0, 1) for _ in VALUES],
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

## Recap

- A `Problem` is `generate` + `evaluate` + optional `feasible` + `direction`.
- `Direction.MINIMIZE` is the default. Use `Direction.MAXIMIZE` for profits.
- `better`, `best_of` and `oriented` compare values the way the heuristics do.

Next: [handle constraints](constraints.md).
