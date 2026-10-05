from random import Random

import pytest

from pymetaheuristics.core import Problem, max_iterations
from pymetaheuristics.genetic_algorithm import genetic_algorithm
from pymetaheuristics.simulated_annealing import simulated_annealing


def problem(seen):
    def generate(rng):
        seen.append(rng)
        return rng.sample(range(6), 6)
    return Problem(generate=generate, evaluate=lambda s: s[0] - s[-1])


@pytest.mark.parametrize('heuristic', [genetic_algorithm, simulated_annealing])
def test_generate_uses_run_rng_and_is_reproducible(heuristic):
    seen = []
    rng = Random(7)
    a = heuristic(problem(seen), stop=max_iterations(5), rng=rng)
    assert seen and all(r is rng for r in seen)
    b = heuristic(problem([]), stop=max_iterations(5), rng=7)
    assert (a.best_solution, a.best_value) == (b.best_solution, b.best_value)
