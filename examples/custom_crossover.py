"""Custom GA crossover: order crossover (OX) for permutations (TSP).

Contract: crossover(parent1, parent2, rng=, **kwargs) -> (child1, child2).
"""
from pymetaheuristics.benchmarks import tsp
from pymetaheuristics.core import max_iterations
from pymetaheuristics.genetic_algorithm import genetic_algorithm


def _ox(a, b, i, j):
    """Keep a[i:j], fill the rest with b's cities in b's order."""
    kept = set(a[i:j])
    rest = iter(c for c in b if c not in kept)
    return [a[k] if i <= k < j else next(rest) for k in range(len(a))]


def order_crossover(g1, g2, rng, **kwargs):
    i, j = sorted(rng.sample(range(len(g1) + 1), 2))
    return _ox(g1, g2, i, j), _ox(g2, g1, i, j)


def main(generations=40, seed=2):
    cities = [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2], [1, 2], [0, 2], [0, 1]]
    problem = tsp(cities)
    result = genetic_algorithm(
        problem, stop=max_iterations(generations), rng=seed,
        population_size=20, crossover=order_crossover)
    print('OX GA tour:', result.best_solution, round(result.best_value, 3))
    return problem, result


if __name__ == '__main__':
    main()
