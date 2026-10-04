"""The contract every heuristic satisfies: a plain function, no base class.

Algorithm-specific knobs (population size, cooling schedule, operators...)
are extra keyword-only parameters with defaults, which keeps any such
function structurally compatible with ``Heuristic``.
"""
from random import Random
from typing import Optional, Protocol, Union

from pymetaheuristics.core.problem import Problem
from pymetaheuristics.core.result import OptimizationResult
from pymetaheuristics.core.termination import Stop


class Heuristic(Protocol):
    def __call__(
        self, problem: Problem, *, stop: Stop,
        rng: Optional[Union[Random, int]] = None,
    ) -> OptimizationResult:
        ...  # pragma: no cover
