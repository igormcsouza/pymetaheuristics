from random import Random

from pymetaheuristics.core import Direction, best_of, better
from pymetaheuristics.genetic_algorithm.model import GeneticAlgorithm
from pymetaheuristics.genetic_algorithm.steps.selections import (
    random_weighted_selection)

MIN, MAX = Direction.MINIMIZE, Direction.MAXIMIZE


def test_better_and_best_of_both_directions():
    assert better(1, 2, MIN) and not better(2, 1, MIN)
    assert better(2, 1, MAX) and not better(1, 2, MAX)
    assert not better(1, 1, MIN) and not better(1, 1, MAX)
    assert best_of([3, 1, 2], MIN) == 1
    assert best_of([3, 1, 2], MAX) == 3


def test_selection_prefers_best_for_each_direction():
    pop = [[0], [1], [100]]
    for direction, want, worst in ((MIN, [0], [100]), (MAX, [100], [0])):
        picked = random_weighted_selection(
            pop, lambda g: g[0], k=200, rng=Random(1), direction=direction)
        assert picked.count(want) > 3 * picked.count(worst)


def test_train_optimizes_in_both_directions():
    def run(direction):
        rng = Random(7)
        ga = GeneticAlgorithm(
            fitness_function=sum,
            genome_generator=lambda: [rng.randint(0, 10) for _ in range(2)],
            direction=direction)
        return ga.train(30, 20, rng=3, k=4)[1]

    assert run(MIN) <= 4
    assert run(MAX) >= 16


def test_maximize_respects_constraints():
    rng = Random(7)
    ga = GeneticAlgorithm(
        fitness_function=sum,
        genome_generator=lambda: [rng.randint(0, 10) for _ in range(2)],
        constraints=[lambda g: sum(g) <= 12], direction=MAX)
    best = ga.train(30, 20, rng=3, k=4)[1]
    assert 10 <= best <= 12
