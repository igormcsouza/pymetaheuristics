"""Continuous box-constrained functions.

Representation: list of floats, one per dimension, each inside `bounds`.
Objective: minimize Sphere or Rastrigin (global optimum 0 at the origin).
Constraint: every coordinate within (lower, upper) (`feasible`).
generate() returns a uniform random point in the box.
"""
from math import cos, pi
from typing import Callable, List, Tuple

from pymetaheuristics.core.problem import Direction, Problem


def sphere_fn(x: List[float]) -> float:
    return sum(v * v for v in x)


def rastrigin_fn(x: List[float]) -> float:
    return 10 * len(x) + sum(v * v - 10 * cos(2 * pi * v) for v in x)


def continuous(fn: Callable[[List[float]], float], dim: int,
               bounds: Tuple[float, float]) -> Problem:
    low, high = bounds
    return Problem(
        generate=lambda rng: [rng.uniform(low, high) for _ in range(dim)],
        evaluate=fn,
        feasible=lambda x: len(x) == dim and all(low <= v <= high for v in x),
        direction=Direction.MINIMIZE)
