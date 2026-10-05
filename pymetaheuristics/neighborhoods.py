"""Neighborhoods: ``neighbor(solution, rng) -> new solution``.

Shared across families (SA moves, GA mutations); they never mutate their
input.
"""
from random import Random
from typing import Any, Callable, Sequence

Neighbor = Callable[[Any, Random], Any]


def swap_neighbor(solution: Sequence, rng: Random) -> list:
    """Permutation move: swap two random positions."""
    new = list(solution)
    i, j = rng.sample(range(len(new)), 2)
    new[i], new[j] = new[j], new[i]
    return new


def two_opt_neighbor(solution: Sequence, rng: Random) -> list:
    """Permutation move: reverse a random segment (2-opt)."""
    i, j = sorted(rng.sample(range(len(solution)), 2))
    return (list(solution[:i]) + list(solution[i:j + 1])[::-1]
            + list(solution[j + 1:]))


def bit_flip_neighbor(solution: Sequence, rng: Random) -> list:
    """Binary move: flip one random bit."""
    new = list(solution)
    i = rng.randrange(len(new))
    new[i] = 1 - new[i]
    return new


def gaussian_neighbor(solution: Sequence, rng: Random,
                      sigma: float = 0.1) -> list:
    """Continuous move: add N(0, sigma) to one random coordinate. Bounds are
    left to ``Problem.feasible`` (infeasible moves are redrawn)."""
    new = list(solution)
    new[rng.randrange(len(new))] += rng.gauss(0, sigma)
    return new
