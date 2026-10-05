"""Symmetric Euclidean Traveling Salesman Problem.

Representation: permutation (list) of city indices; the tour is closed.
Objective: minimize total closed tour length (utils.distances).
Constraint: the solution must be a permutation of all cities (`feasible`).
generate() returns a random permutation.
"""
from typing import List

from pymetaheuristics.core.problem import Direction, Problem
from pymetaheuristics.utils.distances import euclidean_distance


def tsp(cities: List[List[float]]) -> Problem:
    n = len(cities)

    def evaluate(tour):
        return sum(
            euclidean_distance(
                cities[tour[position - 1]], cities[tour[position]])
            for position in range(n))

    return Problem(
        generate=lambda rng: rng.sample(range(n), n),
        evaluate=evaluate,
        feasible=lambda tour: sorted(tour) == list(range(n)),
        direction=Direction.MINIMIZE)
