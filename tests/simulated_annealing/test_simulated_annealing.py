from random import Random

import pytest

from pymetaheuristics.core import (
    Direction, Heuristic, Problem, max_evaluations, max_iterations)
from pymetaheuristics.simulated_annealing import (
    bit_flip_neighbor, geometric_cooling, linear_cooling,
    simulated_annealing, swap_neighbor, two_opt_neighbor)

WEIGHTS, VALUES, CAP = [3, 4, 5, 9], [4, 5, 7, 10], 9


def knapsack(direction=Direction.MAXIMIZE):
    return Problem(
        generate=lambda: [0, 0, 0, 0],
        evaluate=lambda s: sum(v * b for v, b in zip(VALUES, s)),
        feasible=lambda s: sum(w * b for w, b in zip(WEIGHTS, s)) <= CAP,
        direction=direction)


DIST = [[0, 1, 5, 4], [1, 0, 2, 6], [5, 2, 0, 3], [4, 6, 3, 0]]


def tsp(direction=Direction.MINIMIZE):
    def length(t):
        return sum(DIST[t[i - 1]][t[i]] for i in range(len(t)))
    return Problem(generate=lambda: [0, 2, 1, 3], evaluate=length,
                   direction=direction)


def test_knapsack_maximize():
    r = simulated_annealing(
        knapsack(), stop=max_iterations(300), rng=1,
        neighbor=bit_flip_neighbor, initial_temperature=5)
    assert sum(w * b for w, b in zip(WEIGHTS, r.best_solution)) <= CAP
    assert r.best_value == 12  # items 1 and 2


def test_knapsack_minimize_stays_feasible_and_improves():
    r = simulated_annealing(
        knapsack(Direction.MINIMIZE), stop=max_iterations(100), rng=1,
        neighbor=bit_flip_neighbor, initial_temperature=5)
    assert r.best_value == 0 and r.best_solution == [0, 0, 0, 0]


@pytest.mark.parametrize('neighbor', [swap_neighbor, two_opt_neighbor])
@pytest.mark.parametrize('direction', list(Direction))
def test_tsp_both_directions(neighbor, direction):
    r = simulated_annealing(
        tsp(direction), stop=max_iterations(200), rng=2, neighbor=neighbor,
        initial_temperature=3)
    assert sorted(r.best_solution) == [0, 1, 2, 3]
    assert r.best_value == (
        10 if direction is Direction.MINIMIZE else 17)


def test_protocol_result_and_determinism():
    h: Heuristic = simulated_annealing
    a = h(tsp(), stop=max_iterations(50), rng=5)
    b = simulated_annealing(tsp(), stop=max_iterations(50), rng=Random(5))
    assert a.history == b.history and a.best_solution == b.best_solution
    assert a.iterations == 50 and len(a.history) == 51
    assert a.history == sorted(a.history, reverse=True)
    assert a.metadata['termination'] == 'stop'
    assert a.metadata['evaluations'] == 51
    assert a.metadata['final_temperature'] == pytest.approx(
        100 * 0.95 ** 50)


def test_max_evaluations_stop():
    r = simulated_annealing(tsp(), stop=max_evaluations(10), rng=1)
    assert r.metadata['evaluations'] == 10


def test_zero_temperature_is_greedy():
    r = simulated_annealing(
        tsp(), stop=max_iterations(30), rng=3, initial_temperature=0)
    assert r.best_value == 10


def test_infeasible_neighbours_are_bounded():
    p = Problem(generate=lambda: [0, 0], evaluate=lambda s: sum(s),
                feasible=lambda s: s == [0, 0])
    r = simulated_annealing(
        p, stop=max_iterations(5), rng=1, neighbor=bit_flip_neighbor,
        max_neighbor_tries=3)
    assert r.best_solution == [0, 0] and r.metadata['evaluations'] == 1


def test_neighborhoods_do_not_mutate():
    rng = Random(0)
    for n, s in [(swap_neighbor, [0, 1, 2, 3]),
                 (two_opt_neighbor, [0, 1, 2, 3]),
                 (bit_flip_neighbor, [0, 1, 0, 1])]:
        orig = list(s)
        n(s, rng)
        assert s == orig


def test_cooling():
    assert geometric_cooling(0.5)(8) == 4
    assert linear_cooling(3)(8) == 5 and linear_cooling(3)(2) == 0
