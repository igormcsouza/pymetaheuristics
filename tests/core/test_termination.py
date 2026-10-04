from pymetaheuristics.core.problem import Direction
from pymetaheuristics.core.termination import (
    State, any_of, max_evaluations, max_iterations, max_time, target_value)


def state(iteration=0, evaluations=0, elapsed=0.0, best_value=10.0):
    return State(iteration, evaluations, elapsed, best_value)


def test_budgets():
    assert not max_iterations(3)(state(iteration=2))
    assert max_iterations(3)(state(iteration=3))
    assert not max_evaluations(5)(state(evaluations=4))
    assert max_evaluations(5)(state(evaluations=5))
    assert not max_time(1.0)(state(elapsed=0.5))
    assert max_time(1.0)(state(elapsed=1.0))


def test_target_value_respects_direction():
    assert target_value(10.0)(state(best_value=9.0))
    assert not target_value(10.0)(state(best_value=11.0))
    hit = target_value(10.0, Direction.MAXIMIZE)
    assert hit(state(best_value=11.0))
    assert not hit(state(best_value=9.0))


def test_any_of():
    stop = any_of(max_iterations(3), target_value(0.0))
    assert not stop(state(iteration=1, best_value=5.0))
    assert stop(state(iteration=3, best_value=5.0))
    assert stop(state(iteration=1, best_value=0.0))
    assert not any_of()(state())
