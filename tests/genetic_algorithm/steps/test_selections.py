from pymetaheuristics.genetic_algorithm.steps.selections import (
    random_weighted_selection)


def test_genetic_algorithm_steps_selections_random_weighted_selection():

    population = [
        [0, 1, 1, 0, 1, 0, 0],
        [1, 1, 1, 0, 1, 0, 1],
        [0, 1, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 1],
    ]

    best = random_weighted_selection(
        population=population,
        fitness_function=lambda x: sum([i*x for i, x in enumerate(x)]),
        k=2
    )

    assert len(best) == 2
    for i in range(2):
        assert best[i] in population


def test_random_weighted_selection_is_scale_invariant():
    from collections import Counter
    from random import Random

    population = [[0], [1], [2], [3]]

    def picks(scale):
        chosen = random_weighted_selection(
            population, lambda g: g[0] * scale, k=4000, rng=Random(0))
        return Counter(g[0] for g in chosen)

    small = picks(1e-6)  # range << 1 used to give near-uniform weights
    assert small == picks(1e6)
    assert small[0] > 2 * small[2] > 0 and small[3] > 0


def test_random_weighted_selection_equal_values_is_uniform():
    from random import Random

    chosen = random_weighted_selection(
        [[0], [1]], lambda g: 5.0, k=1000, rng=Random(0))
    assert 400 < sum(g[0] for g in chosen) < 600
