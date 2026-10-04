"""Pure helpers so every heuristic honours Direction in one place."""
from typing import Iterable

from pymetaheuristics.core.problem import Direction


def oriented(value: float, direction: Direction) -> float:
    """Value where lower is always better."""
    return -value if direction is Direction.MAXIMIZE else value


def better(value: float, other: float, direction: Direction) -> bool:
    """True if a is strictly better than b."""
    return oriented(value, direction) < oriented(other, direction)


def best_of(values: Iterable[float], direction: Direction) -> float:
    return min(values, key=lambda v: oriented(v, direction))
