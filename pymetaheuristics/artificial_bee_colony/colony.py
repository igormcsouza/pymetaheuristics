"""Artificial Bee Colony as a plain function over the core API."""
from pymetaheuristics.core import (
    InfeasibleError, OptimizationResult, Problem, Stop, better,
    make_rng, oriented, reject, run)
from pymetaheuristics.neighborhoods import Neighbor, swap_neighbor


def artificial_bee_colony(
    problem: Problem, *, stop: Stop, rng=None,
    colony_size: int = 20, limit: int = 50,
    neighbor: Neighbor = swap_neighbor,
    max_neighbor_tries: int = 100,
    max_start_tries: int = 1000,
) -> OptimizationResult:
    """Colony of ``colony_size`` food sources (solutions). One iteration:

    1. employed bees: each source tries one neighbour, kept if better;
    2. onlooker bees: ``colony_size`` times, pick a source by binary
       tournament (works for any value sign), try a neighbour, keep if better;
    3. scouts: a source that failed to improve more than ``limit`` times is
       replaced by a fresh ``problem.generate()`` solution.

    Infeasible neighbours are redrawn up to ``max_neighbor_tries`` times,
    else the attempt counts as a failure. ``history`` is the best value after
    each iteration (first entry: the initial colony); ``metadata`` has
    evaluations, termination ('stop') and state. Each iteration costs about
    ``2 * colony_size`` evaluations (more when scouts fire).
    """
    rng = make_rng(rng)
    direction = problem.direction
    feasible_neighbor = reject(neighbor, problem.feasible, max_neighbor_tries)
    # the only place that calls generate (its signature is changing)
    new_source = reject(lambda: problem.generate(), problem.feasible,
                        max_start_tries)

    def summary(sources, values):
        i = min(range(len(values)), key=lambda k: oriented(values[k],
                                                           direction))
        return sources[i], values[i]

    def init(problem):
        sources = [new_source() for _ in range(colony_size)]
        values = [problem.evaluate(s) for s in sources]
        best, best_value = summary(sources, values)
        return (sources, values, [0] * colony_size), best, best_value, None

    def step(problem, carry):
        sources, values, trials = carry

        def try_improve(i):
            try:
                candidate = feasible_neighbor(sources[i], rng)
            except InfeasibleError:
                trials[i] += 1
                return
            value = problem.evaluate(candidate)
            if better(value, values[i], direction):
                sources[i], values[i], trials[i] = candidate, value, 0
            else:
                trials[i] += 1

        for i in range(colony_size):  # employed
            try_improve(i)
        for _ in range(colony_size):  # onlookers
            a, b = rng.randrange(colony_size), rng.randrange(colony_size)
            try_improve(a if not better(values[b], values[a], direction)
                        else b)
        for i in range(colony_size):  # scouts
            if trials[i] > limit:
                sources[i] = new_source()
                values[i], trials[i] = problem.evaluate(sources[i]), 0
        best, best_value = summary(sources, values)
        return (sources, values, trials), best, best_value, None

    return run(problem, stop=stop, init=init, step=step)
