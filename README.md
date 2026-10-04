# Pymetaheuristics

Combinatorial Optimization problems with quickly good soving.

[![Continuous Integration](https://github.com/igormcsouza/pymetaheuristics/actions/workflows/integration.yml/badge.svg)](https://github.com/igormcsouza/pymetaheuristics/actions/workflows/integration.yml)
[![Coverage Status](https://coveralls.io/repos/github/igormcsouza/pymetaheuristics/badge.svg?branch=master)](https://coveralls.io/github/igormcsouza/pymetaheuristics?branch=master)
[![PyPI version](https://badge.fury.io/py/pymetaheuristics.svg)](https://badge.fury.io/py/pymetaheuristics)


## Introduction

Pymetaheuristics is a package to help build and train Metaheuristics to solve
real world problems mathematically modeled. It strives to generalize the
overall idea of the technic and delivers to the user a friendly wrapper so the
cientist may focus on the problem modeling rather than the heuristic
implementation. This package is an open source project so feel free to send
your implementations and fixes so they may be helpful for others too.

## Requires

Only need **Python>=3.7**. For now, no additional packages will be used.

## Subpackages

The idea is to implement all possible Metaheuristics found on the market today
and some helper functions to improve what is already there.
**Note: This package is under construction, new features will come up soon.**

What Metaheuristics can be found on this project?

1. Genetic Algorithm

## How to use

First install the package (available on pypi)
```bash
$ pip install pymetaheuristics
```
Requires Python 3.12+. For development use [uv](https://docs.astral.sh/uv/):
`uv sync`, then `uv run pre-commit install`.
Lint with `uv run ruff check .` and test with `sh scripts/test.sh`.

Describe your problem with a `Problem` and pass it to a heuristic, which
returns an `OptimizationResult`.
```python
from pymetaheuristics.core import Direction, Problem, max_iterations
from pymetaheuristics.genetic_algorithm import genetic_algorithm
from pymetaheuristics.genetic_algorithm.steps.crossovers import (
    pmx_single_point)

problem = Problem(generate=genome_generator, evaluate=fitness_function,
                  feasible=constraint, direction=Direction.MINIMIZE)

result = genetic_algorithm(
    problem, stop=max_iterations(15), rng=42, population_size=10,
    crossover=pmx_single_point)
result.best_solution, result.best_value, result.history
```
The `GeneticAlgorithm` class still works but is deprecated: it is a thin
wrapper around `genetic_algorithm`.

Every module has its integration test, which I submit the model for testing
with very know NP-Hard problems today (Knapsack, tsp, ...). If you want to see
how it goes, check out the integrations under the model testing folder.

## Reproducibility

Pass `rng=` (an `int` seed or a `random.Random`) to a heuristic and it is
handed to the selection, crossover and mutation steps, so the same seed gives
the same result. Custom steps receive `rng` as a keyword argument. Limitation:
the problem's `generate` is called without an rng, so seed whatever
random source it uses yourself (e.g. a seeded `random.Random` in a closure).

## Adding a heuristic

Heuristics are plain functions `heuristic(problem, *, stop, rng=None, ...)`
returning an `OptimizationResult`. See
[docs/architecture.md](docs/architecture.md).

## Examples

Runnable scripts in [examples/](examples/) extend the library from outside,
using only the public API (each is also run by `tests/examples`):

- [custom_selection.py](examples/custom_selection.py): tournament selection
- [custom_crossover.py](examples/custom_crossover.py): order crossover (TSP)
- [custom_mutation_neighborhood.py](examples/custom_mutation_neighborhood.py):
  insertion move as GA mutation and SA neighbor
- [custom_constraint_repair.py](examples/custom_constraint_repair.py):
  knapsack feasibility by repair and by penalty
- [custom_heuristic.py](examples/custom_heuristic.py): a new heuristic
  (hill climbing with restarts) as a plain function

## How to contribute

Your code and help is very appreciate! Please, send your issue and pr's 
whenever is good for you! If needed, send an 
[email](mailto:igormcsouza@gmail.com) to me I'll be very glad to help. Let's 
build up together.
## Operator ownership

Selection, crossover and mutation operators never modify their inputs: they
return new genomes (`new = mutate(genome)`), so callers may keep using the
originals safely.
