# Pymetaheuristics

Combinatorial Optimization problems with quickly good soving.

[![Continuous Integration](https://github.com/igormcsouza/pymetaheuristics/actions/workflows/integration.yml/badge.svg)](https://github.com/igormcsouza/pymetaheuristics/actions/workflows/integration.yml)
[![Coverage Status](https://coveralls.io/repos/github/igormcsouza/pymetaheuristics/badge.svg?branch=master)](https://coveralls.io/github/igormcsouza/pymetaheuristics?branch=master)


## Introduction

Pymetaheuristics is a package to help build and train Metaheuristics to solve
real world problems mathematically modeled. It strives to generalize the
overall idea of the technic and delivers to the user a friendly wrapper so the
cientist may focus on the problem modeling rather than the heuristic
implementation. This package is an open source project so feel free to send
your implementations and fixes so they may be helpful for others too.


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

Import the algorithm model you want to use to solve you problem. Implement the
needed functions and pass to the model. Train and get the results.
```python
from pymetaheuristics.genetic_algorithm.model import GeneticAlgorithm

model = GeneticAlgorithm(
    fitness_function=fitness_function,
    genome_generator=genome_generator
)

result = model.train(
    epochs=15, pop_size=10, crossover=pmx_single_point, verbose=True)
```

Every module has its integration test, which I submit the model for testing
with very know NP-Hard problems today (Knapsack, tsp, ...). If you want to see
how it goes, check out the integrations under the model testing folder.

## How to contribute

Your code and help is very appreciate! Please, send your issue and pr's 
whenever is good for you! If needed, send an 
[email](mailto:igormcsouza@gmail.com) to me I'll be very glad to help. Let's 
build up together.