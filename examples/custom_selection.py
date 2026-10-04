"""Custom GA selection: tournament selection, passed as ``selection=``.

Contract: selection(population, fitness, rng=, direction=, **kwargs)
returns the parents (the GA breeds the first two).
"""
from pymetaheuristics.benchmarks import knapsack
from pymetaheuristics.core import Direction, max_iterations, oriented
from pymetaheuristics.genetic_algorithm import genetic_algorithm


def tournament_selection(population, fitness, rng, direction=None, k=2,
                         size=3, **kwargs):
    """Return k winners; each wins a tournament among `size` random picks."""
    direction = direction or Direction.MINIMIZE
    winners = []
    for _ in range(k):
        contenders = rng.sample(population, min(size, len(population)))
        best = min(contenders, key=lambda g: oriented(fitness(g), direction))
        winners.append(list(best))  # copy: operators must not alias
    return winners


def main(generations=30, seed=1):
    problem = knapsack([60, 100, 120, 80, 30], [10, 20, 30, 25, 5], 50, seed)
    result = genetic_algorithm(
        problem, stop=max_iterations(generations), rng=seed,
        selection=tournament_selection)
    print('tournament GA:', result.best_solution, result.best_value)
    return problem, result


if __name__ == '__main__':
    main()
