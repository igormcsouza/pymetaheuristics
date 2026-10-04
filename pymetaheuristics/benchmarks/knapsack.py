"""0/1 Knapsack.

Representation: list of 0/1 ints, one per item (1 = packed).
Objective: maximize total value of packed items.
Constraint: total weight of packed items <= capacity (`feasible`).
generate() returns a random feasible packing.
"""
from typing import List, Optional, Union
from random import Random

from pymetaheuristics.core.problem import Direction, Problem
from pymetaheuristics.utils.rng import make_rng


def knapsack(values: List[float], weights: List[float], capacity: float,
             rng: Optional[Union[Random, int]] = None) -> Problem:
    rng = make_rng(rng)
    n = len(values)

    def total(solution, data):
        return sum(d for d, bit in zip(data, solution) if bit)

    def generate():
        solution = [0] * n
        for i in rng.sample(range(n), n):  # random order, add while it fits
            solution[i] = 1
            if total(solution, weights) > capacity:
                solution[i] = 0
        return solution

    return Problem(
        generate=generate,
        evaluate=lambda s: total(s, values),
        feasible=lambda s: len(s) == n and total(s, weights) <= capacity,
        direction=Direction.MAXIMIZE)
