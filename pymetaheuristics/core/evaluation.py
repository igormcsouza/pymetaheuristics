"""Objective-evaluation counting without touching algorithm code."""
from dataclasses import replace
from typing import Callable, Tuple

from pymetaheuristics.core.problem import Problem


def counting(problem: Problem) -> Tuple[Problem, Callable[[], int]]:
    """Return a copy of ``problem`` whose ``evaluate`` is counted, plus a
    zero-arg function reading the count so far."""
    calls = 0

    def evaluate(solution):
        nonlocal calls
        calls += 1
        return problem.evaluate(solution)

    return replace(problem, evaluate=evaluate), lambda: calls
