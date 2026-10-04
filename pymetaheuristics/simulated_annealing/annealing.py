"""Simulated Annealing as a plain function over the core API."""
import math

from pymetaheuristics.core import (
    InfeasibleError, OptimizationResult, Problem, Stop, make_rng, oriented,
    reject, run)
from pymetaheuristics.simulated_annealing.cooling import (
    Cooling, geometric_cooling)
from pymetaheuristics.simulated_annealing.neighborhoods import (
    Neighbor, swap_neighbor)


def simulated_annealing(
    problem: Problem, *, stop: Stop, rng=None,
    neighbor: Neighbor = swap_neighbor,
    initial_temperature: float = 100.0,
    cooling: Cooling = geometric_cooling(0.95),
    max_neighbor_tries: int = 100,
    max_start_tries: int = 1000,
) -> OptimizationResult:
    """Metropolis acceptance: a worse move of oriented size d is accepted
    with probability exp(-d / T). The start is drawn up to
    ``max_start_tries`` times for a feasible one. Infeasible neighbours are
    redrawn up to ``max_neighbor_tries`` times, else the iteration keeps the
    current solution. ``history`` is the best value after each iteration
    (first entry: the start); ``metadata`` has evaluations,
    final_temperature, termination ('stop') and state (the final State).
    """
    rng = make_rng(rng)
    direction = problem.direction
    feasible_neighbor = reject(neighbor, problem.feasible, max_neighbor_tries)

    def init(problem):
        current = reject(problem.generate, problem.feasible,
                         max_start_tries)()
        value = problem.evaluate(current)
        return (current, value, initial_temperature), current, value, None

    def step(problem, carry):
        current, current_value, temperature = carry
        try:
            candidate = feasible_neighbor(current, rng)
        except InfeasibleError:
            pass  # keep the current solution this iteration
        else:
            value = problem.evaluate(candidate)
            delta = (oriented(value, direction)
                     - oriented(current_value, direction))
            if delta <= 0 or (
                    temperature > 0
                    and rng.random() < math.exp(-delta / temperature)):
                current, current_value = candidate, value
        carry = current, current_value, cooling(temperature)
        return carry, current, current_value, None

    return run(problem, stop=stop, init=init, step=step,
               extras=lambda carry: {'final_temperature': carry[2]})
