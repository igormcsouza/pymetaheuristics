import random
from dataclasses import FrozenInstanceError

import pytest

from pymetaheuristics.core import Direction, Problem
from pymetaheuristics.utils.distances import euclidean_distance


def test_defaults():
    p = Problem(generate=lambda rng: 1, evaluate=lambda s: s)

    assert p.direction is Direction.MINIMIZE
    assert p.feasible('anything') is True


def test_frozen():
    p = Problem(generate=lambda rng: 1, evaluate=lambda s: s)

    with pytest.raises(FrozenInstanceError):
        p.direction = Direction.MAXIMIZE


def knapsack_problem():
    values = [10, 5, 8]
    weights = [4, 3, 5]
    return Problem(
        generate=lambda rng: [rng.randint(0, 1) for _ in values],
        evaluate=lambda s: sum(v * x for v, x in zip(values, s)),
        feasible=lambda s: sum(w * x for w, x in zip(weights, s)) <= 8,
        direction=Direction.MAXIMIZE,
    )


def tsp_problem():
    cities = [[0., 0.], [0., 3.], [4., 3.]]

    def evaluate(tour):
        legs = zip(tour, tour[1:] + tour[:1])
        return sum(euclidean_distance(cities[a], cities[b])
                   for a, b in legs)

    return Problem(
        generate=lambda rng: rng.sample(range(len(cities)), len(cities)),
        evaluate=evaluate,
    )


def test_knapsack():
    p = knapsack_problem()

    assert len(p.generate(random.Random(0))) == 3
    assert p.evaluate([1, 1, 0]) == 15
    assert p.feasible([1, 1, 0])
    assert not p.feasible([1, 0, 1])
    assert p.direction is Direction.MAXIMIZE


def test_tsp():
    p = tsp_problem()

    assert sorted(p.generate(random.Random(0))) == [0, 1, 2]
    assert p.evaluate([0, 1, 2]) == 12.
    assert p.feasible([0, 1, 2])
    assert p.direction is Direction.MINIMIZE


@pytest.mark.parametrize('make', [knapsack_problem, tsp_problem])
def test_same_api(make):
    p = make()
    s = p.generate(random.Random(0))

    assert isinstance(p.evaluate(s), (int, float))
    assert isinstance(p.feasible(s), bool)
