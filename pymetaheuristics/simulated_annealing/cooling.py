"""Cooling schedules: plain functions ``temperature -> next temperature``."""
from typing import Callable

Cooling = Callable[[float], float]


def geometric_cooling(alpha: float = 0.95) -> Cooling:
    """T <- alpha * T, with 0 < alpha < 1."""
    return lambda temperature: temperature * alpha


def linear_cooling(step: float) -> Cooling:
    """T <- max(T - step, 0)."""
    return lambda temperature: max(temperature - step, 0.0)
