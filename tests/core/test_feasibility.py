import pytest

from pymetaheuristics.core import (
    Direction, InfeasibleError, Problem, penalty, reject, repair)


def test_reject_retries_until_feasible():
    values = iter([1, 3, 4])
    gen = reject(lambda: next(values), lambda x: x % 2 == 0)
    assert gen() == 4


def test_reject_bounded_retries():
    calls = []
    gen = reject(lambda: calls.append(1) or 1, lambda x: False, max_tries=5)
    with pytest.raises(InfeasibleError):
        gen()
    assert len(calls) == 5


def test_reject_passes_arguments():
    op = reject(lambda x, k=0: x + k, lambda x: x > 0)
    assert op(1, k=1) == 2


def test_repair_fixes_solution():
    gen = repair(lambda: -3, abs, feasible=lambda x: x >= 0)
    assert gen() == 3


def test_repair_still_infeasible_raises():
    gen = repair(lambda: 1, lambda x: x, feasible=lambda x: False)
    with pytest.raises(InfeasibleError):
        gen()


def test_penalty_minimize_and_maximize():
    def pen(x):
        return max(0, x - 10)

    for direction, bad in ((Direction.MINIMIZE, 14), (Direction.MAXIMIZE, 10)):
        p = Problem(lambda: 0, lambda x: x, lambda x: x <= 10, direction)
        q = penalty(p, pen)
        assert q.evaluate(5) == 5
        assert q.evaluate(12) == bad
        assert q.feasible(12) is True
        assert p.feasible(12) is False


def test_penalty_composes_with_replace_for_generate():
    from dataclasses import replace
    p = Problem(lambda: 0, lambda x: x, lambda x: x <= 10, Direction.MINIMIZE)
    q = replace(penalty(p, lambda x: max(0, x - 10)), generate=lambda rng: 7)
    assert q.generate(None) == 7
    assert q.evaluate(12) == 14
