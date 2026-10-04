"""Benchmark record shared by every problem in the suite.

Heuristic-agnostic: a Benchmark only wraps a core Problem plus reference data.
Expected evaluation criterion: run any heuristic for a fixed budget and report
`gap(benchmark, value)` of the best feasible value found (0 = optimal),
optionally with the number of evaluations needed to reach `known_best`.
"""
from dataclasses import dataclass
from typing import Any, Optional

from pymetaheuristics.core.problem import Problem


@dataclass(frozen=True)
class Benchmark:
    name: str
    problem: Problem
    known_best: Optional[float]      # None when no reference is known
    known_best_solution: Any = None  # a feasible solution reaching it
    source: str = ''                 # how known_best was established
    criteria: str = 'gap to known_best of the best feasible value found'


def gap(benchmark: Benchmark, value: float) -> float:
    """Non-negative gap to the known best: |value - best| / max(|best|, 1).

    Relative for large optima, absolute for optima near 0 (e.g. Sphere).
    """
    best = benchmark.known_best
    if best is None:
        raise ValueError("%s has no known best" % benchmark.name)
    return abs(value - best) / max(abs(best), 1.0)
