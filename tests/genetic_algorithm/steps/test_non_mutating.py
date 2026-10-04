import copy

import pytest

from pymetaheuristics.genetic_algorithm.steps.crossovers import (
    pmx_single_point, single_point_crossover)
from pymetaheuristics.genetic_algorithm.steps.multations import inter_mutation
from pymetaheuristics.genetic_algorithm.steps.selections import (
    random_weighted_selection)


@pytest.mark.parametrize("cross", [single_point_crossover, pmx_single_point])
@pytest.mark.parametrize("g1,g2", [
    ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0]), ([0], [0])])
def test_crossover_does_not_mutate(cross, g1, g2):
    o1, o2 = copy.deepcopy(g1), copy.deepcopy(g2)
    c1, c2 = cross(g1, g2)
    assert g1 == o1 and g2 == o2
    assert c1 is not g1 and c1 is not g2
    assert c2 is not g1 and c2 is not g2


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_inter_mutation_does_not_mutate(probability):
    g = [0, 1, 2, 3, 4, 5]
    out = inter_mutation(g, q=5, probability=probability)
    assert g == [0, 1, 2, 3, 4, 5]
    assert out is not g


def test_selection_does_not_mutate():
    pop = [[1, 2], [3, 4], [5, 6]]
    orig = copy.deepcopy(pop)
    out = random_weighted_selection(pop, lambda g: -sum(g), k=3)
    assert pop == orig
    assert all(o is not p for o in out for p in pop)
