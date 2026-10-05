"""Handling a knapsack capacity constraint two ways, no library changes.

- repair: ``repair=`` fixes infeasible GA children (drop items until fit);
- penalty: ``penalty()`` wraps the problem so infeasible packings are
  allowed but score worse.
"""
from dataclasses import replace

from pymetaheuristics.benchmarks import knapsack
from pymetaheuristics.core import max_iterations, penalty
from pymetaheuristics.genetic_algorithm import genetic_algorithm

VALUES = [60, 100, 120, 80, 30, 45]
WEIGHTS = [10, 20, 30, 25, 5, 15]
CAPACITY = 50


def weight(packing):
    return sum(w for w, bit in zip(WEIGHTS, packing) if bit)


def drop_heaviest(packing):
    """Repair: unpack the heaviest packed item until the load fits."""
    packing = list(packing)
    while weight(packing) > CAPACITY:
        packed = [i for i, bit in enumerate(packing) if bit]
        packing[max(packed, key=lambda i: WEIGHTS[i])] = 0
    return packing


def main(generations=30, seed=4):
    problem = knapsack(VALUES, WEIGHTS, CAPACITY)
    stop = max_iterations(generations)
    repaired = genetic_algorithm(
        problem, stop=stop, rng=seed, repair=drop_heaviest)
    # penalty: 10 value units per unit of overweight; any bit string is ok
    soft = penalty(
        replace(problem, generate=lambda rng: [
            rng.randint(0, 1) for _ in VALUES]),
        lambda s: 10 * max(0, weight(s) - CAPACITY))
    penalized = genetic_algorithm(soft, stop=stop, rng=seed)
    print('repair  :', repaired.best_solution, repaired.best_value)
    print('penalty :', penalized.best_solution, penalized.best_value)
    return problem, repaired, penalized, soft


if __name__ == '__main__':
    main()
