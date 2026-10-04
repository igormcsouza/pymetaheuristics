"""Composable stop conditions: a ``Stop`` is a predicate over ``State``."""
from dataclasses import dataclass
from typing import Callable

from pymetaheuristics.core.problem import Direction


@dataclass(frozen=True)
class State:
    """Snapshot of a run, built by the heuristic once per iteration."""
    iteration: int
    evaluations: int
    elapsed: float
    best_value: float


Stop = Callable[[State], bool]


def max_iterations(n: int) -> Stop:
    return lambda s: s.iteration >= n


def max_evaluations(n: int) -> Stop:
    return lambda s: s.evaluations >= n


def max_time(seconds: float) -> Stop:
    return lambda s: s.elapsed >= seconds


def target_value(
    value: float, direction: Direction = Direction.MINIMIZE
) -> Stop:
    # ponytail: inline comparison; switch to core.direction helpers (#21)
    if direction is Direction.MINIMIZE:
        return lambda s: s.best_value <= value
    return lambda s: s.best_value >= value


def any_of(*stops: Stop) -> Stop:
    return lambda s: any(stop(s) for stop in stops)
