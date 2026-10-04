"""Composable stop conditions: a ``Stop`` is a predicate over ``State``."""
from dataclasses import dataclass
from typing import Callable

from pymetaheuristics.core.direction import better
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
    return lambda state: state.iteration >= n


def max_evaluations(n: int) -> Stop:
    return lambda state: state.evaluations >= n


def max_time(seconds: float) -> Stop:
    return lambda state: state.elapsed >= seconds


def target_value(
    value: float, direction: Direction = Direction.MINIMIZE
) -> Stop:
    return lambda state: not better(value, state.best_value, direction)


def any_of(*stops: Stop) -> Stop:
    return lambda state: any(stop(state) for stop in stops)
