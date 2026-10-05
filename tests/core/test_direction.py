from random import Random

from pymetaheuristics.core import Direction, best_of, better, oriented
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
        scores = [oriented(g[0], direction) for g in pop]
        picked = random_weighted_selection(
            pop, scores, Random(1), k=200)
        assert picked.count(want) > 3 * picked.count(worst)
