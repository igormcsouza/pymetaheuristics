from pymetaheuristics.benchmarks import gap, get
from pymetaheuristics.core import max_iterations
from pymetaheuristics.genetic_algorithm import genetic_algorithm
from pymetaheuristics.genetic_algorithm.steps.crossovers import (
    pmx_single_point)


def test_ga_knapsack_close_to_optimum():
    bench = get('knapsack-10')
    result = genetic_algorithm(
        bench.problem, stop=max_iterations(30), rng=0, population_size=10)
    assert bench.problem.feasible(result.best_solution)
    assert gap(bench, result.best_value) <= 0.3


def test_ga_tsp_close_to_optimum():
    bench = get('tsp-grid9')
    result = genetic_algorithm(
        bench.problem, stop=max_iterations(30), rng=0, population_size=10,
        crossover=pmx_single_point)
    assert sorted(result.best_solution) == list(range(9))
    assert gap(bench, result.best_value) <= 0.5
