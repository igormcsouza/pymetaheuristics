"""Custom GA selection: tournament selection, passed as ``selection=``.

Contract: selection(population, scores, rng) returns the parents (the GA
breeds the first two). Scores are oriented: lower is better, in both
directions. Bind knobs such as k with functools.partial.
"""
from pymetaheuristics.benchmarks import knapsack
from pymetaheuristics.core import max_iterations
from pymetaheuristics.genetic_algorithm import genetic_algorithm


def tournament_selection(population, scores, rng, k=2, size=3):
    """Return k winners; each wins a tournament among `size` random picks."""
    winners = []
    for _ in range(k):
        contenders = rng.sample(
            range(len(population)), min(size, len(population)))
        best = min(contenders, key=scores.__getitem__)
        winners.append(list(population[best]))  # copy: operators must not alias
    return winners


def main(generations=30, seed=1):
    problem = knapsack([60, 100, 120, 80, 30], [10, 20, 30, 25, 5], 50)
    result = genetic_algorithm(
        problem, stop=max_iterations(generations), rng=seed,
        selection=tournament_selection)
    print('tournament GA:', result.best_solution, result.best_value)
    return problem, result


if __name__ == '__main__':
    main()
