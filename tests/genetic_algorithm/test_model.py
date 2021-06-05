from pymetaheuristics.genetic_algorithm.exceptions import LoadHistoryException
from pymetaheuristics.genetic_algorithm.types import GeneticAlgorithmHistory
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

    assert ga_model._check_constraints([1, 2, 3, 4, 6]) is True

    result, score = ga_model.train(1, 10, verbose=True)

    assert len(result) == 5
    assert sum(result) == score
    assert 5 not in result
    assert 0 not in result


def test_genetic_algorithm_model_load_history():
    ga_model = GeneticAlgorithm(
        fitness_function=lambda x: sum(x),
        genome_generator=lambda: [randint(0, 10) for _ in range(5)]
    )

    history: GeneticAlgorithmHistory = {
        0: {
            "runs": [([0, 1, 2], 3.0)],
            "best": ([0, 1, 2], 3.0),
            "elapsed": 0.00001,
            "args": {
                "epochs": 1,
                "pop_size": 10,
                "selection": "selection_method",
                "crossover": "crossover_method",
                "mutation": "mutation_method",
                "verbose": True,
                "kwargs": {"k": 2}
            }
        }
    }

    ga_model.load_history(history)

    assert len(ga_model.history.keys()) == 1


def test_genetic_algorithm_model_load_history_failed():
    ga_model = GeneticAlgorithm(
        fitness_function=lambda x: sum(x),
        genome_generator=lambda: [randint(0, 10) for _ in range(5)]
    )

    history1 = None
    history2 = {"0": {"wrong": "args"}}

    try:
        ga_model.load_history(history1)  # type: ignore
    except LoadHistoryException:
        assert True
    else:
        assert False

    try:
        ga_model.load_history(history2)  # type: ignore
    except LoadHistoryException:
        assert True
    else:
        assert False
