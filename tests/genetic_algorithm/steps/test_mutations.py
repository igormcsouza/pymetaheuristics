from pymetaheuristics.genetic_algorithm.steps.mutations import inter_mutation


def test_genetic_algorithm_steps_mutations_inter_mutation():

    genome = [0, 1, 1, 0, 1, 0, 0]

    mutated = inter_mutation(genome, num_swaps=3, probability=0.4)

    assert len(mutated) == len(genome)
    assert sum(mutated) == sum(genome)


def test_legacy_names_still_work():
    from pymetaheuristics.genetic_algorithm.steps import multations
    from pymetaheuristics.utils.distances import (
        euclidean_distance, euclidian_distance)

    assert multations.inter_mutation is inter_mutation
    assert euclidian_distance is euclidean_distance
    genome = [0, 1, 2, 3]
    assert sorted(inter_mutation(genome, q=3, probability=1.0)) == genome
