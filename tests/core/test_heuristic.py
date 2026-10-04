"""End-to-end check of the extension API with a reference hill climber."""
from time import perf_counter

from pymetaheuristics.core import (
    Direction, Heuristic, OptimizationResult, Problem, State, any_of,
    counting, max_evaluations, max_iterations, target_value)
from pymetaheuristics.utils.rng import make_rng


def hill_climb(problem, *, stop, rng=None, step=3):
    rng = make_rng(rng)
    problem, evaluations = counting(problem)
    sign = 1 if problem.direction is Direction.MINIMIZE else -1
    start = perf_counter()
    best = problem.generate()
    best_value = problem.evaluate(best)
    history, iteration = [best_value], 0
    while True:
        state = State(iteration, evaluations(), perf_counter() - start,
                      best_value)
        if stop(state):
            break
        candidate = best + rng.randint(-step, step)
        value = problem.evaluate(candidate)
        if sign * value < sign * best_value:
            best, best_value = candidate, value
        iteration += 1
        history.append(best_value)
    return OptimizationResult(
        best, best_value, history=history, iterations=iteration,
        elapsed=state.elapsed, metadata={'evaluations': state.evaluations})


def make_problem(direction=Direction.MINIMIZE):
    sign = 1 if direction is Direction.MINIMIZE else -1
    return Problem(generate=lambda: 100,
                   evaluate=lambda x: sign * (x - 7) ** 2,
                   direction=direction)


def test_reference_heuristic_satisfies_protocol():
    heuristic: Heuristic = hill_climb
    result = heuristic(make_problem(), stop=max_iterations(20), rng=1)
    assert result.iterations == 20
    assert len(result.history) == 21
    assert result.metadata['evaluations'] == 21
    assert result.history == sorted(result.history, reverse=True)


def test_same_seed_same_result():
    a = hill_climb(make_problem(), stop=max_iterations(30), rng=7)
    b = hill_climb(make_problem(), stop=max_iterations(30), rng=7)
    assert a.best_solution == b.best_solution
    assert a.history == b.history


def test_composed_stop_and_direction():
    result = hill_climb(
        make_problem(Direction.MAXIMIZE),
        stop=any_of(max_evaluations(10_000),
                    target_value(0, Direction.MAXIMIZE)),
        rng=3)
    assert result.best_solution == 7
    assert result.metadata['evaluations'] < 10_000
