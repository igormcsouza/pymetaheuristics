"""A new heuristic, hill climbing with restarts, as a plain function.

It satisfies the ``Heuristic`` contract (problem, *, stop, rng) and reuses
the core: ``counting`` for evaluations, ``State``/``stop`` for termination,
``better`` for direction, and ``OptimizationResult`` for the return value.
"""
from time import perf_counter

from pymetaheuristics.benchmarks import tsp
from pymetaheuristics.core import (
    OptimizationResult, State, better, counting, max_evaluations)
from pymetaheuristics.simulated_annealing import two_opt_neighbor
from pymetaheuristics.utils.rng import make_rng


def restart_hill_climbing(problem, *, stop, rng=None,
                          neighbor=two_opt_neighbor, patience=20):
    """Climb with `neighbor`; restart from a fresh solution when stuck."""
    rng = make_rng(rng)
    problem, evaluations = counting(problem)
    start, direction = perf_counter(), problem.direction
    current = problem.generate()
    current_value = problem.evaluate(current)
    best, best_value = current, current_value
    history, iteration, stuck = [best_value], 0, 0
    while True:
        state = State(iteration, evaluations(), perf_counter() - start,
                      best_value)
        if stop(state):
            break
        candidate = neighbor(current, rng)
        if problem.feasible(candidate):
            value = problem.evaluate(candidate)
            if better(value, current_value, direction):
                current, current_value, stuck = candidate, value, 0
            else:
                stuck += 1
        if stuck >= patience:
            current = problem.generate()
            current_value, stuck = problem.evaluate(current), 0
        if better(current_value, best_value, direction):
            best, best_value = current, current_value
        iteration += 1
        history.append(best_value)
    return OptimizationResult(
        best, best_value, history=history, iterations=iteration,
        elapsed=state.elapsed,
        metadata={'evaluations': state.evaluations, 'termination': state})


def main(evaluations=500, seed=5):
    cities = [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2], [1, 2], [0, 2], [0, 1]]
    problem = tsp(cities, seed)
    result = restart_hill_climbing(
        problem, stop=max_evaluations(evaluations), rng=seed)
    print('restart hill climbing:', round(result.best_value, 3),
          'in', result.metadata['evaluations'], 'evaluations')
    return problem, result


if __name__ == '__main__':
    main()
