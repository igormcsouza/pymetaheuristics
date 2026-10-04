"""Functional feasibility strategies; Problem.feasible is the contract.

Each strategy wraps a function that produces solutions (a generator or a
non-mutating operator), or a Problem, so no strategy is hard-coded.
"""
from dataclasses import replace
from typing import Any, Callable, Optional

from pymetaheuristics.core.direction import oriented
from pymetaheuristics.core.problem import Problem


class InfeasibleError(RuntimeError):
    """A strategy could not produce a feasible solution."""


def reject(
    produce: Callable[..., Any], feasible: Callable[[Any], bool],
    max_tries: int = 1000
) -> Callable[..., Any]:
    """Call `produce` until feasible, at most `max_tries` times."""
    def wrapped(*args, **kwargs):
        for _ in range(max_tries):
            solution = produce(*args, **kwargs)
            if feasible(solution):
                return solution
        raise InfeasibleError(
            "no feasible solution after %i tries" % max_tries)
    return wrapped


def repair(
    produce: Callable[..., Any], fix: Callable[[Any], Any],
    feasible: Optional[Callable[[Any], bool]] = None
) -> Callable[..., Any]:
    """Pass every produced solution through `fix`; if `feasible` is given,
    raise InfeasibleError when the repaired solution is still infeasible."""
    def wrapped(*args, **kwargs):
        solution = fix(produce(*args, **kwargs))
        if feasible is not None and not feasible(solution):
            raise InfeasibleError("repair did not give a feasible solution")
        return solution
    return wrapped


def penalty(
    problem: Problem, penalty_fn: Callable[[Any], float]
) -> Problem:
    """Problem whose evaluate is worsened by `penalty_fn(solution)` (>= 0,
    zero when feasible); infeasible solutions are then allowed.

    `generate` is intentionally left unchanged. To also change it, compose:
    ``dataclasses.replace(penalty(p, f), generate=g)``.
    """
    def evaluate(solution):
        return problem.evaluate(solution) + oriented(
            penalty_fn(solution), problem.direction)

    return replace(problem, evaluate=evaluate, feasible=lambda s: True)
