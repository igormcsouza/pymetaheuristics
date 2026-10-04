# Extension architecture

How a new heuristic (Simulated Annealing #11, a refactored GA #25, ...) plugs
into pymetaheuristics without touching the core problem model. The guiding
rule (umbrella #12): a functional core. Heuristics are plain functions, not
subclasses, and nothing is a class that merely wraps a function.

```
pymetaheuristics/core/
  problem.py      Problem, Direction        what is being optimized
  result.py       OptimizationResult        what every run returns
  termination.py  State, Stop, stop helpers  when a run ends
  evaluation.py   counting()                how evaluations are counted
  heuristic.py    Heuristic (Protocol)      the contract tying it together
  loop.py         run()                     the shared iteration loop
utils/rng.py      make_rng()                reproducibility (re-exported by core)
```

## 1. What interface must every heuristic satisfy?

A heuristic is a function:

```python
def heuristic(problem: Problem, *, stop: Stop, rng=None,
              **algorithm_knobs) -> OptimizationResult
```

`core.heuristic.Heuristic` is a `typing.Protocol` describing exactly that,
used for type checking only. There is no base class to inherit and nothing to
register: any function with this shape is a heuristic.

- `problem` is the only view of the domain (`generate`, `evaluate`,
  `feasible`, `direction`). A heuristic never imports problem-specific code.
- `stop` and `rng` are keyword-only and common to all heuristics.
- Algorithm-specific knobs (population size, temperature schedule, operators)
  are extra keyword-only parameters with defaults. Defaults keep the function
  compatible with the Protocol.

Sketch of Simulated Annealing under this contract:

```python
def simulated_annealing(problem, *, stop, rng=None, neighbor=swap_neighbor,
                        temperature=exponential_cooling(100, 0.95)):
    ...
    return OptimizationResult(best, best_value, history=..., ...)
```

## 2. What is shared between heuristics?

Only data and small functions, all in `core`:

| Shared piece | Where |
|---|---|
| Problem model and optimization direction | `core/problem.py` |
| Result shape | `core/result.py` |
| Stop conditions | `core/termination.py` |
| Evaluation counting | `core/evaluation.py` |
| Randomness (`int` seed or `random.Random`) | `utils/rng.py` |
| Comparing values under a direction | `core/direction.py` (#21) |
| Handling infeasible solutions | `core/feasibility.py` (#24) |

Operators that are reusable across families (e.g. a swap neighbor used by
both a GA mutation and an SA move) live as plain functions in a shared module
(`pymetaheuristics/neighborhoods.py`)
and are passed in as keyword arguments. Operators never mutate their inputs.

## 3. What belongs to an algorithm vs the core engine?

There is no engine object. What differs between families (generations vs a
single trajectory vs restarts) is one iteration, so an algorithm writes two
plain functions and hands them to `core.loop.run`:

```python
init(problem)        -> (carry, solution, value, record)
step(problem, carry) -> (carry, solution, value, record)

return run(problem, stop=stop, init=init, step=step,
           extras=lambda carry: {...})   # optional extra metadata
```

`carry` is whatever the algorithm threads between iterations (current
solution, population, temperature...). `solution`/`value` is the
iteration's best candidate. `record` is the iteration's `history` entry;
`None` records the best value so far.

- **Core (`run`):** evaluation counting, the clock, building `State`,
  checking `stop` before each step, best-so-far under `problem.direction`,
  `history` and the `OptimizationResult`. It knows nothing about
  populations, temperatures or neighborhoods.
- **Algorithm:** `init`, `step` and its operators.

`simulated_annealing` and `genetic_algorithm` are both built on `run` (and
so are `examples/custom_heuristic.py` and the experiments' random search). A
heuristic may still write its own loop (see the hill climber under
*Reference implementation*), as long as it satisfies the Protocol.

## 4. How are termination conditions represented?

A `Stop` is a predicate `Callable[[State], bool]`. `State` is a frozen
snapshot (`iteration`, `evaluations`, `elapsed`, `best_value`, `direction`)
built once per iteration (by `run`, or by a hand-written loop). `direction`
defaults to `MINIMIZE`; `run` fills it from the problem, so
`target_value(0.0)` compares in the problem's direction (pass a direction to
override). Helpers build stops, and `any_of` composes them:

```python
from pymetaheuristics.core import (
    any_of, max_evaluations, max_iterations, max_time, target_value)

stop = any_of(max_iterations(500), max_evaluations(10_000), max_time(2.0),
              target_value(0.0))
```

Users may pass any function of `State`, e.g. a stagnation check that
closes over its own memory. `all_of` was skipped; add it when someone needs
it. Heuristics built on `run` report `metadata['termination'] == 'stop'`
and the final `State` (the one that satisfied `stop`) as
`metadata['state']`; inspect it to see which budget was hit.

## 5. How are objective evaluations counted?

`counting(problem)` returns a copy of the problem whose `evaluate` counts
calls, plus a zero-argument function that reads the count:

```python
problem, evaluations = counting(problem)
...
State(iteration, evaluations(), elapsed, best_value)
```

Algorithms use the counted problem exactly like the original, so counting
never leaks into operator code, and the user's `Problem` is left untouched.
The count feeds `max_evaluations` and can be reported in the result's
`metadata`.

## 6. How do algorithms expose history and intermediate results?

Through the returned `OptimizationResult`:

- `history` is a list with one record per iteration. A best-value float is
  the default; an algorithm may record a dict of stats instead (e.g. GA mean
  fitness, SA temperature) and documents its shape. A dict record should
  include the best value so far under `'best_so_far'` (as the GA does), so
  convergence curves compare directly with float histories such as SA's.
- `iterations`, `elapsed` and `metadata` cover the rest. Under `run`,
  `metadata` always has `evaluations`, `termination` (`'stop'`) and `state`
  (final `State`), plus whatever the algorithm's `extras(carry)` returns
  (e.g. SA's `final_temperature`).

Live progress callbacks were deliberately left out: the result already holds
everything, and a `Stop` sees every `State`. Add an `on_iteration` keyword
when a real use case (progress bars, plotting during a run) shows up.

## Reference implementation

`tests/core/test_heuristic.py` contains a ~25-line hill climber written
against this API. It exercises the Protocol, `counting`, composed stops,
both directions and seeded reproducibility. It deliberately writes its own
loop to document the bare Protocol; for a new heuristic, prefer `run` and
copy `simulated_annealing` in
`pymetaheuristics/simulated_annealing/annealing.py` (an `init`/`step` pair)
or `examples/custom_heuristic.py`.

For a population-based heuristic, `genetic_algorithm` in
`pymetaheuristics/genetic_algorithm/algorithm.py` is the full reference: an
`init`/`step` pair on `run` (the carry is the population, its values and the
elite), one small private helper per generation step, feasibility enforced
before evaluation, and a dict of stats per generation in `history`.
