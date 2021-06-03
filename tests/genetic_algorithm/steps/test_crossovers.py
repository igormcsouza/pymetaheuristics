from pymetaheuristics.genetic_algorithm.steps import (
    single_point_crossover, pmx_single_point
)


def test_genetic_algorithm_steps_crossovers_single_point_crossover():

    parent_g1 = [0, 0, 1, 0, 1, 0, 0]
    parent_g2 = [1, 1, 0, 0, 1, 1, 1]

    g1, g2 = single_point_crossover(parent_g1, parent_g2)

    for i, (gene1, gene2) in enumerate(zip(g1, g2)):
        assert gene1 == parent_g1[i] or gene1 == parent_g2[i]
        assert gene2 == parent_g1[i] or gene2 == parent_g2[i]


def test_genetic_algorithm_steps_crossovers_pmx_single_point():

    parent_g1 = [1, 2, 3, 4, 5]
    parent_g2 = [5, 4, 3, 2, 1]

    g1, g2 = pmx_single_point(parent_g1, parent_g2)

    assert sum(g1) == sum(g2) == 15
