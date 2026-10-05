from pymetaheuristics.artificial_bee_colony import (
    artificial_bee_colony, bit_flip_neighbor, gaussian_neighbor)
from pymetaheuristics.benchmarks.continuous import continuous, sphere_fn
from pymetaheuristics.core import (
    Direction, Heuristic, Problem, max_evaluations, max_iterations)

WEIGHTS, VALUES, CAP = [3, 4, 5, 9], [4, 5, 7, 10], 9


def knapsack(direction=Direction.MAXIMIZE):
    return Problem(
        generate=lambda rng: [0, 0, 0, 0],
        evaluate=lambda s: sum(v * b for v, b in zip(VALUES, s)),
        feasible=lambda s: sum(w * b for w, b in zip(WEIGHTS, s)) <= CAP,
        direction=direction)


def test_knapsack_maximize():
    r = artificial_bee_colony(
        knapsack(), stop=max_iterations(30), rng=1, colony_size=6,
        limit=5, neighbor=bit_flip_neighbor)
    assert r.best_value == 12


def test_knapsack_minimize_respects_direction():
    r = artificial_bee_colony(
        knapsack(Direction.MINIMIZE), stop=max_iterations(10), rng=1,
        colony_size=4, neighbor=bit_flip_neighbor)
    assert r.best_value == 0


def test_tsp_permutation():
    p = Problem(generate=lambda rng: rng.sample(range(6), 6),
                evaluate=lambda t: sum(abs(t[i] - t[i - 1])
                                       for i in range(6)))
    r = artificial_bee_colony(p, stop=max_iterations(20), rng=2)
    assert sorted(r.best_solution) == list(range(6))
    assert r.best_value == 10


def test_continuous_beats_random_and_is_deterministic():
    def go(seed, stop):
        return artificial_bee_colony(
            continuous(sphere_fn, 3, (-5, 5)), stop=stop,
            rng=seed, neighbor=gaussian_neighbor)
    r = go(0, max_iterations(100))
    assert r.best_value < 0.1
    assert r.history == go(0, max_iterations(100)).history
    assert r.history == sorted(r.history, reverse=True)
    assert r.best_value < go(0, max_iterations(0)).best_value


def test_protocol_and_stop():
    h: Heuristic = artificial_bee_colony
    r = h(knapsack(), stop=max_evaluations(10), rng=1)
    assert r.metadata['termination'] == 'stop'
    assert r.metadata['evaluations'] >= 10
    r = artificial_bee_colony(knapsack(), stop=max_iterations(3), rng=1)
    assert r.iterations == 3 and len(r.history) == 4


def test_scouts_replace_stale_sources():
    gen = iter(range(1000))
    p = Problem(generate=lambda rng: [next(gen)], evaluate=lambda s: 0.0)
    artificial_bee_colony(p, stop=max_iterations(3), rng=1, colony_size=2,
                          limit=0, neighbor=lambda s, rng: s)
    assert next(gen) > 2  # constant objective: scouts fire every iteration


def test_infeasible_neighbors_count_as_failed_trials():
    p = Problem(generate=lambda rng: [0], evaluate=lambda s: 0.0,
                feasible=lambda s: s == [0])
    r = artificial_bee_colony(
        p, stop=max_iterations(2), rng=1, colony_size=2,
        neighbor=lambda s, rng: [1], max_neighbor_tries=1)
    assert r.best_solution == [0]
