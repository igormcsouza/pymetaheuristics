from itertools import permutations, product

import pytest

from pymetaheuristics.benchmarks import (
    BENCHMARKS, all_benchmarks, gap, get)
from pymetaheuristics.core import Direction

ALL = all_benchmarks()


def ids(bs):
    return [b.name for b in bs]


def test_registry():
    assert set(BENCHMARKS) == {b.name for b in ALL}
    assert get('knapsack-3') is BENCHMARKS['knapsack-3']
    with pytest.raises(KeyError):
        get('nope')


@pytest.mark.parametrize('b', ALL, ids=ids(ALL))
def test_known_best_solution(b):
    p = b.problem
    assert p.feasible(b.known_best_solution)
    assert p.evaluate(b.known_best_solution) == pytest.approx(b.known_best)
    assert gap(b, b.known_best) == 0


@pytest.mark.parametrize('b', ALL, ids=ids(ALL))
def test_generate_feasible_and_not_better_than_best(b):
    for _ in range(50):
        s = b.problem.generate()
        assert b.problem.feasible(s)
        v = b.problem.evaluate(s)
        if b.problem.direction is Direction.MAXIMIZE:
            assert v <= b.known_best + 1e-9
        else:
            assert v >= b.known_best - 1e-9


def test_directions():
    for b in ALL:
        expected = (Direction.MAXIMIZE if b.name.startswith('knapsack')
                    else Direction.MINIMIZE)
        assert b.problem.direction is expected


def test_infeasible_detected():
    assert not get('knapsack-3').problem.feasible([1, 1, 1])
    assert not get('tsp-ring8').problem.feasible([0] * 8)
    assert not get('sphere-5').problem.feasible([9.0] * 5)


def test_gap_requires_known_best():
    b = get('sphere-5')
    assert gap(b, 0.5) == 0.5
    with pytest.raises(ValueError):
        gap(type(b)('x', b.problem, None), 1.0)


def test_brute_force_knapsack():
    for name in ('knapsack-3', 'knapsack-10'):
        b = get(name)
        n = len(b.known_best_solution)
        best = max(b.problem.evaluate(list(s))
                   for s in product((0, 1), repeat=n)
                   if b.problem.feasible(list(s)))
        assert best == b.known_best


def test_brute_force_tsp():
    for name in ('tsp-ring8', 'tsp-grid9'):
        b = get(name)
        n = len(b.known_best_solution)
        best = min(b.problem.evaluate([0] + list(r))
                   for r in permutations(range(1, n)))
        assert best == pytest.approx(b.known_best)
