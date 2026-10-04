"""Custom GA mutation and SA neighbor: the same insertion move.

GA contract:  mutation(genome, rng=, **kwargs) -> new genome.
SA contract:  neighbor(solution, rng) -> new solution.
Neither may mutate its input.
"""
from pymetaheuristics.benchmarks import tsp
from pymetaheuristics.core import max_iterations
from pymetaheuristics.genetic_algorithm import genetic_algorithm
from pymetaheuristics.simulated_annealing import simulated_annealing


def insertion_neighbor(solution, rng):
    """Remove one element and re-insert it at another position."""
    new = list(solution)
    new.insert(rng.randrange(len(new)), new.pop(rng.randrange(len(new))))
    return new


def insertion_mutation(genome, rng, **kwargs):
    return insertion_neighbor(genome, rng)


def main(iterations=200, seed=3):
    cities = [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2], [1, 2], [0, 2], [0, 1]]
    problem = tsp(cities, seed)
    ga = genetic_algorithm(
        problem, stop=max_iterations(iterations // 5), rng=seed,
        mutation=insertion_mutation)
    sa = simulated_annealing(
        problem, stop=max_iterations(iterations), rng=seed,
        neighbor=insertion_neighbor)
    print('GA insertion mutation:', round(ga.best_value, 3))
    print('SA insertion neighbor:', round(sa.best_value, 3))
    return problem, ga, sa


if __name__ == '__main__':
    main()
