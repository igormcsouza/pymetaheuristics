from pymetaheuristics.genetic_algorithm.steps import inter_mutation


def test_genetic_algorithm_steps_multations_inter_mutation():

    g = [0, 1, 1, 0, 1, 0, 0]

    g_changed = inter_mutation(g, q=3, probability=0.4)

    assert len(g_changed) == len(g)
    assert sum(g_changed) == sum(g)
