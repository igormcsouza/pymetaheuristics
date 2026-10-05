# Release notes

What each release contained, what broke, and how to upgrade. The short
version lives in the [changelog](https://github.com/igormcsouza/pymetaheuristics/blob/main/CHANGELOG.md).

## 0.3.0 (2026-10-05)

Removes everything deprecated in 0.2 and changes the operator and
`Problem.generate` contracts, so operators draw from the run's `rng`.

!!! warning "Breaking changes"
    - **`Problem.generate` takes the run's `rng`.**
    - **GA operators take `rng`** and selection gets oriented scores; extra
      knobs are bound with `functools.partial`, there is no `**operator_kwargs`.
    - **Removed** the deprecated names listed below.

New in 0.3.0: `artificial_bee_colony()` and the `gaussian_neighbor` move.

### Upgrading from 0.2

Problem generation:

```text
# 0.2
problem = Problem(generate=lambda: rng.sample(range(5), 5), evaluate=f)

# 0.3
problem = Problem(generate=lambda rng: rng.sample(range(5), 5), evaluate=f)
```

Operators (`rng` is required and positional; selection receives *scores*
oriented so that lower is always better, whatever the direction):

```text
# 0.2
def mutation(genome, rng=None, **kwargs): ...
def crossover(parent1, parent2, rng=None, **kwargs): ...
def selection(population, fitness_function, k=2, rng=None,
              direction=MINIMIZE, **kwargs): ...
genetic_algorithm(problem, stop=stop, mutation=inter_mutation, q=3)

# 0.3
def mutation(genome, rng, ...): ...
def crossover(parent1, parent2, rng): ...
def selection(population, scores, rng, k): ...  # the GA asks for k=2
genetic_algorithm(problem, stop=stop,
                  mutation=partial(inter_mutation, num_swaps=3))
```

Benchmark factories no longer take an `rng`/`seed`: `tsp(cities, 0)` becomes
`tsp(cities)`, likewise `knapsack(...)` and `continuous(...)`. Seed the
heuristic instead (`rng=`).

`inter_mutation` is now `inter_mutation(genome, rng, num_swaps=2,
probability=0.75)`.

Removed names:

| 0.2 (deprecated) | 0.3 |
|---|---|
| `GeneticAlgorithm(...)`, `.train()` | `genetic_algorithm(problem, stop=..., ...)` |
| `GeneticAlgorithmHistory`, `LoadHistoryException` | `result.history` (a list of dicts) |
| `genetic_algorithm.steps.multations` | `genetic_algorithm.steps.mutations` |
| `utils.distances.euclidian_distance` | `utils.distances.euclidean_distance` (or `math.dist`) |
| `inter_mutation(genome, q=...)` | `inter_mutation(genome, rng, num_swaps=...)` |
| neighborhoods imported from the simulated annealing module | `pymetaheuristics.neighborhoods` |

## 0.2.0 (2026-10-04)

A rewrite of the core around plain functions. Python 3.12+.

!!! warning "Breaking changes"
    Nothing from 0.1 was removed, but these changed behaviour:

    - **Python 3.12+ is required** (0.1.x supported 3.7+).
    - **Operators no longer modify their inputs.** Crossover, mutation and
      selection return new genomes. Code that relied on in-place changes
      must use the return value.
    - **Selection weights changed.** 0.1 used `-fitness` as the weight, which
      only worked for negative fitness values. Weights are now scaled by the
      value range, so any objective scale works and the worst genome is
      still selectable. Runs differ from 0.1 even with the same seed.
    - **The GA breeds the whole population** every generation. In 0.1 most
      of each generation was refilled with random genomes, so results are
      much better, and different.
    - **The GA respects `Direction`** in selection and best tracking. In 0.1
      it always minimized, and you negated profits yourself.
    - **Exceptions changed base class.** `CrossOverException` is now a
      `ValueError` and `LoadHistoryException` an `Exception`. In 0.1 both
      were `BaseException`, so a plain `except Exception` did not catch them
      and now does.
    - **`euclidean_distance`** is `math.dist`, so it raises `ValueError`
      instead of `AssertionError` on a length mismatch.

!!! info "Deprecated, removal planned for 0.3"
    Tracked in [#50](https://github.com/igormcsouza/pymetaheuristics/issues/50).
    These still work but emit a `DeprecationWarning`:

    - the `GeneticAlgorithm` class (now a wrapper over `genetic_algorithm()`),
    - `genetic_algorithm.steps.multations` (use `mutations`),
    - `utils.distances.euclidian_distance` (use `euclidean_distance`),
    - `inter_mutation(q=...)` (use `num_swaps=`).

New in 0.2.0:

- `Problem`, `Direction` and `OptimizationResult`, and the reference loop
  `core.run` with stopping criteria (`max_iterations`, `max_evaluations`,
  `max_time`, `target_value`, `any_of`).
- `genetic_algorithm()` and `simulated_annealing()` as plain functions with
  pluggable operators (`swap_neighbor`, `two_opt_neighbor`,
  `bit_flip_neighbor`, `geometric_cooling`, `linear_cooling`).
- A single `feasible` predicate, with `reject`, `repair` and `penalty`
  helpers.
- Reproducible runs through `rng=`.
- The [benchmark suite](benchmarks.md), the [experiments](experiments.md),
  runnable [extension examples](extending.md) and this documentation site.
- A release workflow that publishes to PyPI and attaches the sdist and wheel.

### Upgrading from 0.1

| 0.1 | Now |
|---|---|
| `GeneticAlgorithm(fitness_function, genome_generator, constraints, direction=...)` | `Problem(generate=..., evaluate=..., feasible=..., direction=...)` |
| `ga.add_constraint(c)` / `constraints=[...]` | one `feasible` predicate: `lambda s: all(c(s) for c in constraints)` |
| `ga.train(epochs, pop_size, ...)` returns `(genome, fitness)` | `genetic_algorithm(problem, stop=max_iterations(epochs), population_size=pop_size, ...)` returns an `OptimizationResult` |
| `ga.history` keyed by timestamp | `result.history`, one dict per generation (see [Results and history](tutorial/results.md)) |
| `verbose=True` | loop over `result.history` after the run |
| `genetic_algorithm.steps.multations` | `genetic_algorithm.steps.mutations` |
| `inter_mutation(genome, q=...)` | `inter_mutation(genome, num_swaps=...)` |
| `utils.distances.euclidian_distance` | `utils.distances.euclidean_distance` (or `math.dist`) |


Before and after:

```text
# 0.1
ga = GeneticAlgorithm(fitness, generate, constraints=[fits])
genome, value = ga.train(epochs=15, pop_size=10, rng=42)
```

```python
from pymetaheuristics.core import Problem, max_iterations
from pymetaheuristics.genetic_algorithm import genetic_algorithm

problem = Problem(generate=lambda rng: rng.sample(range(5), 5),
                  evaluate=lambda s: sum(abs(g - i) for i, g in enumerate(s)))
result = genetic_algorithm(problem, stop=max_iterations(15), rng=42,
                           population_size=10)
genome, value = result.best_solution, result.best_value
```

## 0.1.1 (2021-06-05)

A small feature release on the same `GeneticAlgorithm` class.

- **Added** `GeneticAlgorithm.load_history(history)`, to load a previously
  saved history. It validates the pattern and raises `LoadHistoryException`
  if keys such as `args`, `runs`, `best` or `elapsed` are missing.
- **Added** the `GeneticAlgorithmHistory` type.
- **Changed** each history entry to also hold `best` and `elapsed`.
- **Changed** the supported Python to 3.7+ (was 3.8+), and added a PyPI
  badge and a "Requires" section to the README.

No breaking changes.

## 0.1.0 (2021-06-03)

The first release: a Genetic Algorithm, with no dependencies.

- `GeneticAlgorithm(fitness_function, genome_generator, constraints)`
  with `train(epochs, pop_size, selection, crossover, mutation, verbose,
  **kwargs)` returning `(genome, fitness)`. `add_constraint()` adds a
  constraint, and `history` records runs by timestamp.
- Step functions: `random_weighted_selection`, `single_point_crossover`,
  `pmx_single_point` (for TSP) and `inter_mutation`.
- `euclidian_distance`.
- The GA **only minimized**: to maximize, you returned the negated value.
- Integration tests on Knapsack and TSP, CI and delivery to PyPI.
- Python 3.8+.
