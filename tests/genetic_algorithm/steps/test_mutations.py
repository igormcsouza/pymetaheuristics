from pymetaheuristics.genetic_algorithm.steps.mutations import inter_mutation


def test_genetic_algorithm_steps_mutations_inter_mutation():

    genome = [0, 1, 1, 0, 1, 0, 0]

    mutated = inter_mutation(genome, num_swaps=3, probability=0.4)

    assert len(mutated) == len(genome)
    assert sum(mutated) == sum(genome)
