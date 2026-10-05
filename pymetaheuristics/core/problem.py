from dataclasses import dataclass
from enum import Enum
from random import Random
from typing import Any, Callable


class Direction(Enum):
    MINIMIZE = 'minimize'
    MAXIMIZE = 'maximize'


def _always_feasible(solution: Any) -> bool:
    return True


@dataclass(frozen=True)
class Problem:
    """Domain-agnostic optimization problem; heuristics only use this."""
    generate: Callable[[Random], Any]
    evaluate: Callable[[Any], float]
    feasible: Callable[[Any], bool] = _always_feasible
    direction: Direction = Direction.MINIMIZE
