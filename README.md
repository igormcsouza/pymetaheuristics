# pymetaheuristics

[![Continuous Integration](https://github.com/igormcsouza/pymetaheuristics/actions/workflows/integration.yml/badge.svg)](https://github.com/igormcsouza/pymetaheuristics/actions/workflows/integration.yml)
[![Coverage Status](https://coveralls.io/repos/github/igormcsouza/pymetaheuristics/badge.svg?branch=main)](https://coveralls.io/github/igormcsouza/pymetaheuristics?branch=main)
[![PyPI version](https://badge.fury.io/py/pymetaheuristics.svg)](https://badge.fury.io/py/pymetaheuristics)

Metaheuristics for optimization problems in plain Python, with no
dependencies. Describe the problem as a `Problem`: how to generate a
solution, how to evaluate it, which solutions are feasible, and whether to
minimize or maximize. Then pass it to a heuristic: a Genetic Algorithm or
Simulated Annealing. Every heuristic returns the same `OptimizationResult`.

Documentation: <https://igormcsouza.github.io/pymetaheuristics/>
([changelog](CHANGELOG.md)).

## Install

Requires Python 3.12+.

```sh
pip install pymetaheuristics
# or
uv add pymetaheuristics
```

## Quickstart

```python
from random import Random

from pymetaheuristics.core import Direction, Problem, max_iterations
from pymetaheuristics.genetic_algorithm import genetic_algorithm
from pymetaheuristics.simulated_annealing import (
    bit_flip_neighbor, simulated_annealing)

VALUES, WEIGHTS, CAPACITY = [60, 100, 120], [10, 20, 30], 50
seeded = Random(0)

knapsack = Problem(
    generate=lambda: [seeded.randint(0, 1) for _ in VALUES],
    evaluate=lambda s: sum(v for v, bit in zip(VALUES, s) if bit),
    feasible=lambda s: sum(w for w, bit in zip(WEIGHTS, s) if bit)
    <= CAPACITY,
    direction=Direction.MAXIMIZE,
)

ga = genetic_algorithm(knapsack, stop=max_iterations(20), rng=42)
sa = simulated_annealing(knapsack, stop=max_iterations(200), rng=42,
                         neighbor=bit_flip_neighbor)
print(ga.best_solution, ga.best_value)  # [0, 1, 1] 220
print(sa.best_solution, sa.best_value)  # [0, 1, 1] 220
```

## Documentation

The documentation lives in [docs/](docs/). Start with [docs/index.md](docs/index.md),
or build the site locally with `uv run --group docs mkdocs serve`. It
covers:

- [Tutorial](docs/tutorial/index.md): problems, directions, constraints, both
  heuristics, results and history.
- [Worked examples](docs/examples.md): Knapsack and TSP.
- [Extending](docs/extending.md): custom operators and heuristics, plus
  the runnable [examples/](examples/).
- [Architecture](docs/architecture.md), [benchmarks](docs/benchmarks.md),
  [experiments](docs/experiments.md).
- [Release notes](docs/release-notes.md): what changed between versions.

## Development

```sh
uv sync                    # dev tools
uv run pre-commit install
uv run ruff check .
sh scripts/test.sh         # pytest with coverage, including the doc snippets
uv run --group docs mkdocs build --strict
```

Contributions are welcome. Open an issue or a pull request.
