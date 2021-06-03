from random import randint

from pymetaheuristics.genetic_algorithm.model import GeneticAlgorithm


def test_genetic_algorithm_model_instantiate():
    ga_model = GeneticAlgorithm(
        fitness_function=lambda x: sum(x),
        genome_generator=lambda: [randint(0, 100) for _ in range(5)],
        constraints=[
            lambda x: 10 in x
        ]
    )

    assert ga_model is not None


def test_genetic_algorithm_model_train():
    ga_model = GeneticAlgorithm(
        fitness_function=lambda x: sum(x),
        genome_generator=lambda: [randint(0, 10) for _ in range(5)],
        constraints=[
            lambda x: 5 not in x
        ]
    )

    result, score = ga_model.train(1, 10)

    assert len(result) == 5
    assert sum(result) == score
    assert 5 not in result


def test_genetic_algorithm_model_add_constraint():
    ga_model = GeneticAlgorithm(
        fitness_function=lambda x: sum(x),
        genome_generator=lambda: [randint(0, 10) for _ in range(5)],
        constraints=[
            lambda x: 5 not in x
        ]
    )

    ga_model.add_constraint(constraint=lambda x: 0 not in x)

    result, score = ga_model.train(1, 10, verbose=True)

    assert len(result) == 5
    assert sum(result) == score
    assert 5 not in result
    assert 0 not in result
