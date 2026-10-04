"""Run every example with a tiny budget: the extension points work with
user code living outside the library."""
from random import Random

from examples import (
    custom_constraint_repair, custom_crossover, custom_heuristic,
    custom_mutation_neighborhood, custom_selection)
from pymetaheuristics.core import OptimizationResult


def check(problem, result):
    assert isinstance(result, OptimizationResult)
    assert problem.feasible(result.best_solution)
    assert problem.evaluate(result.best_solution) == result.best_value


def test_custom_selection():
    check(*custom_selection.main(generations=5))


def test_custom_crossover():
    check(*custom_crossover.main(generations=5))
    a, b = list(range(8)), list(reversed(range(8)))
    for child in custom_crossover.order_crossover(a, b, Random(0)):
        assert sorted(child) == a


def test_custom_mutation_and_neighborhood():
    problem, ga, sa = custom_mutation_neighborhood.main(iterations=20)
    check(problem, ga)
    check(problem, sa)


def test_custom_constraint_repair():
    problem, repaired, penalized, soft = custom_constraint_repair.main(
        generations=5)
    check(problem, repaired)
    assert penalized.best_value == soft.evaluate(penalized.best_solution)


def test_custom_heuristic():
    problem, result = custom_heuristic.main(evaluations=50)
    check(problem, result)
    assert result.metadata['evaluations'] >= 50
    assert len(result.history) == result.iterations + 1
