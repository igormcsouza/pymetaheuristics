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
