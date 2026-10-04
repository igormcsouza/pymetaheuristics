import copy

import pytest

from pymetaheuristics.genetic_algorithm.steps.crossovers import (
    pmx_single_point, single_point_crossover)
from pymetaheuristics.genetic_algorithm.steps.mutations import inter_mutation
from pymetaheuristics.genetic_algorithm.steps.selections import (
    random_weighted_selection)


@pytest.mark.parametrize("cross", [single_point_crossover, pmx_single_point])
@pytest.mark.parametrize("parent1,parent2", [
    ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0]), ([0], [0])])
def test_crossover_does_not_mutate(cross, parent1, parent2):
    orig1, orig2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
    child1, child2 = cross(parent1, parent2)
    assert parent1 == orig1 and parent2 == orig2
    assert child1 is not parent1 and child1 is not parent2
    assert child2 is not parent1 and child2 is not parent2


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_inter_mutation_does_not_mutate(probability):
    genome = [0, 1, 2, 3, 4, 5]
    out = inter_mutation(genome, num_swaps=5, probability=probability)
    assert genome == [0, 1, 2, 3, 4, 5]
    assert out is not genome


def test_selection_does_not_mutate():
    pop = [[1, 2], [3, 4], [5, 6]]
    orig = copy.deepcopy(pop)
    out = random_weighted_selection(pop, lambda member: -sum(member), k=3)
    assert pop == orig
    assert all(chosen is not member for chosen in out for member in pop)
