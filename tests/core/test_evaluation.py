from pymetaheuristics.core.evaluation import counting
from pymetaheuristics.core.problem import Direction, Problem


def test_counts_and_preserves_problem():
    problem = Problem(
        generate=lambda rng: 1, evaluate=lambda x: x * 2,
        direction=Direction.MAXIMIZE)
    counted, evaluations = counting(problem)
    assert evaluations() == 0
    assert counted.evaluate(3) == 6
    assert counted.evaluate(4) == 8
    assert evaluations() == 2
    assert counted.direction is Direction.MAXIMIZE
    assert counted.generate is problem.generate
    problem.evaluate(1)  # the original stays uncounted
    assert evaluations() == 2
