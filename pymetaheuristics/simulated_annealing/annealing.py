"""Simulated Annealing as a plain function over the core API."""
import math
from time import perf_counter

from pymetaheuristics.core import (
    OptimizationResult, Problem, State, Stop, better, counting, oriented,
    reject)
from pymetaheuristics.simulated_annealing.cooling import (
    Cooling, geometric_cooling)
from pymetaheuristics.neighborhoods import (
    Neighbor, swap_neighbor)
from pymetaheuristics.utils.rng import make_rng


def simulated_annealing(
    problem: Problem, *, stop: Stop, rng=None,
    neighbor: Neighbor = swap_neighbor,
    initial_temperature: float = 100.0,
    cooling: Cooling = geometric_cooling(0.95),
    max_neighbor_tries: int = 100,
) -> OptimizationResult:
    """Metropolis acceptance: a worse move of oriented size d is accepted
    with probability exp(-d / T). Infeasible neighbours are redrawn up to
    ``max_neighbor_tries`` times, else the iteration keeps the current
    solution. ``history`` is the best value after each iteration (first
    entry: the start); ``metadata`` has evaluations, final_temperature and
    termination ('stop').
    """
    rng = make_rng(rng)
    problem, evaluations = counting(problem)
    direction = problem.direction
    start = perf_counter()
    current = reject(problem.generate, problem.feasible)()
    current_value = problem.evaluate(current)
    best, best_value = current, current_value
    temperature = initial_temperature
    history, iteration = [best_value], 0
    while True:
        state = State(iteration, evaluations(), perf_counter() - start,
                      best_value)
        if stop(state):
            break
        for _ in range(max_neighbor_tries):
            candidate = neighbor(current, rng)
            if not problem.feasible(candidate):
                continue
            value = problem.evaluate(candidate)
            delta = (oriented(value, direction)
                     - oriented(current_value, direction))
            if delta <= 0 or (
                    temperature > 0
                    and rng.random() < math.exp(-delta / temperature)):
                current, current_value = candidate, value
                if better(value, best_value, direction):
                    best, best_value = candidate, value
            break
        temperature = cooling(temperature)
        iteration += 1
        history.append(best_value)
    return OptimizationResult(
        best, best_value, history=history, iterations=iteration,
        elapsed=state.elapsed,
        metadata={'evaluations': state.evaluations,
                  'final_temperature': temperature, 'termination': 'stop'})
