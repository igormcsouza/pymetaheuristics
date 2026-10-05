# Stopping

Every heuristic needs a `stop`. It is how you say how much work to spend.

`stop` is required, and it is a predicate over a `State` snapshot
(`iteration`, `evaluations`, `elapsed`, `best_value`, `direction`). Use the
helpers and combine them with `any_of`:

```python
from pymetaheuristics.core import (
    any_of, max_evaluations, max_iterations, max_time, target_value)

stop = any_of(max_iterations(1000), max_evaluations(5000),
              max_time(2.0), target_value(300))
```

!!! note
    `target_value` compares in the problem's direction, so the target `300`
    above means "value at least 300" for this maximization problem.

Any `Callable[[State], bool]` works as a stop. See
[Extending](../extending.md#a-custom-stop) for one.

## Recap

- `stop` is required: a predicate over a `State` snapshot.
- Combine `max_iterations`, `max_evaluations`, `max_time` and `target_value`
  with `any_of`.

Next: [how operators plug in](operators.md).
