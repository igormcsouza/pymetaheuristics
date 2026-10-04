"""The shared run loop: stop checks, counting, timing, best-so-far, result.

A heuristic supplies two plain functions and ``run`` does the rest::

    init(problem)        -> (carry, solution, value, record)
    step(problem, carry) -> (carry, solution, value, record)

``carry`` is whatever the algorithm threads between iterations (a current
solution, a population, a temperature...). ``solution``/``value`` is the
iteration's best candidate; ``run`` keeps the best so far. ``record`` goes
into ``history``; ``None`` records the best value so far.
"""
from time import perf_counter
from typing import Any, Callable, Dict, Optional, Tuple

from pymetaheuristics.core.direction import better
from pymetaheuristics.core.evaluation import counting
from pymetaheuristics.core.problem import Problem
from pymetaheuristics.core.result import OptimizationResult
from pymetaheuristics.core.termination import State, Stop

Outcome = Tuple[Any, Any, float, Any]


def run(
    problem: Problem, *, stop: Stop,
    init: Callable[[Problem], Outcome],
    step: Callable[[Problem, Any], Outcome],
    extras: Optional[Callable[[Any], Dict[str, Any]]] = None,
) -> OptimizationResult:
    """Run ``init`` then ``step`` until ``stop``; both see the counted
    problem. ``stop`` is checked before every step. ``metadata`` holds
    ``evaluations``, ``termination`` ('stop'), ``state`` (the final
    ``State``) and ``extras(final_carry)`` if given."""
    problem, evaluations = counting(problem)
    direction = problem.direction
    start = perf_counter()

    carry, best, best_value, record = init(problem)
    history = [best_value if record is None else record]
    iteration = 0
    while True:
        state = State(iteration, evaluations(), perf_counter() - start,
                      best_value, direction)
        if stop(state):
            break
        carry, solution, value, record = step(problem, carry)
        if better(value, best_value, direction):
            best, best_value = solution, value
        iteration += 1
        history.append(best_value if record is None else record)

    return OptimizationResult(
        best, best_value, history=history, iterations=iteration,
        elapsed=state.elapsed,
        metadata={'evaluations': state.evaluations, 'termination': 'stop',
                  'state': state, **(extras(carry) if extras else {})})
