import importlib

import pytest

from pymetaheuristics.genetic_algorithm.steps.mutations import inter_mutation


def test_genetic_algorithm_steps_mutations_inter_mutation():

    genome = [0, 1, 1, 0, 1, 0, 0]

    mutated = inter_mutation(genome, num_swaps=3, probability=0.4)

    assert len(mutated) == len(genome)
    assert sum(mutated) == sum(genome)


def test_legacy_names_still_work():

    with pytest.warns(DeprecationWarning, match="steps.mutations"):
        multations = importlib.import_module(
            "pymetaheuristics.genetic_algorithm.steps.multations")
    assert multations.inter_mutation is inter_mutation
    genome = [0, 1, 2, 3]
    with pytest.warns(DeprecationWarning, match="num_swaps"):
        mutated = inter_mutation(genome, q=3, probability=1.0)
    assert sorted(mutated) == genome
