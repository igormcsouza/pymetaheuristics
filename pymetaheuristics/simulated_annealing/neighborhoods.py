"""Neighborhoods: ``neighbor(solution, rng) -> new solution``.

They never mutate their input; pass any function of this shape to
``simulated_annealing``.
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
