"""A new heuristic, hill climbing with restarts, as a plain function.

It satisfies the ``Heuristic`` contract (problem, *, stop, rng): it only
says how to start (``init``) and how to take one step (``step``);
``core.run`` does the stop checks, evaluation counting, timing, best-so-far
tracking and builds the ``OptimizationResult``.
"""
from pymetaheuristics.benchmarks import tsp
from pymetaheuristics.core import better, make_rng, max_evaluations, run
from pymetaheuristics.neighborhoods import two_opt_neighbor


def restart_hill_climbing(problem, *, stop, rng=None,
                          neighbor=two_opt_neighbor, patience=20):
    """Climb with `neighbor`; restart from a fresh solution when stuck."""
    rng = make_rng(rng)

    def fresh(problem):
        current = problem.generate(rng)
        return current, problem.evaluate(current), 0

    def init(problem):
        carry = fresh(problem)  # carry: (current, value, stuck steps)
        return carry, carry[0], carry[1], None

    def step(problem, carry):
        current, current_value, stuck = carry
        candidate = neighbor(current, rng)
        if problem.feasible(candidate):
            value = problem.evaluate(candidate)
            if better(value, current_value, problem.direction):
                current, current_value, stuck = candidate, value, 0
            else:
                stuck += 1
        carry = (current, current_value, stuck) if stuck < patience \
            else fresh(problem)
        return carry, carry[0], carry[1], None

    return run(problem, stop=stop, init=init, step=step)


def main(evaluations=500, seed=5):
    cities = [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2], [1, 2], [0, 2], [0, 1]]
    problem = tsp(cities)
    result = restart_hill_climbing(
        problem, stop=max_evaluations(evaluations), rng=seed)
    print('restart hill climbing:', round(result.best_value, 3),
          'in', result.metadata['evaluations'], 'evaluations')
    return problem, result


if __name__ == '__main__':
    main()
