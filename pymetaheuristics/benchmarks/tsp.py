"""Symmetric Euclidean Traveling Salesman Problem.

Representation: permutation (list) of city indices; the tour is closed.
Objective: minimize total closed tour length (utils.distances).
Constraint: the solution must be a permutation of all cities (`feasible`).
generate() returns a random permutation.
"""
from random import Random
from typing import List, Optional, Union

from pymetaheuristics.core.problem import Direction, Problem
from pymetaheuristics.utils.distances import euclidian_distance
from pymetaheuristics.utils.rng import make_rng


def tsp(cities: List[List[float]],
        rng: Optional[Union[Random, int]] = None) -> Problem:
    rng = make_rng(rng)
    n = len(cities)

    def evaluate(tour):
        return sum(euclidian_distance(cities[tour[i - 1]], cities[tour[i]])
                   for i in range(n))

    return Problem(
        generate=lambda: rng.sample(range(n), n),
        evaluate=evaluate,
        feasible=lambda t: sorted(t) == list(range(n)),
        direction=Direction.MINIMIZE)
