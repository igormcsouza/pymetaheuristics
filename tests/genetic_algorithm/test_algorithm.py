from random import Random

import pytest

from pymetaheuristics.core import (
    Direction, Heuristic, InfeasibleError, Problem, State, any_of,
    max_evaluations, max_iterations, oriented, target_value)
from pymetaheuristics.genetic_algorithm import genetic_algorithm


def make_problem(direction=Direction.MINIMIZE, seed=0, feasible=None):
    """Genomes of 6 ints in [0, 9]; objective is their sum."""
    kwargs = {} if feasible is None else {'feasible': feasible}
    return Problem(
        generate=lambda rng: [rng.randint(0, 9) for _ in range(6)],
        evaluate=sum, direction=direction, **kwargs)


def test_satisfies_heuristic_protocol():
    heuristic: Heuristic = genetic_algorithm
    result = heuristic(make_problem(), stop=max_iterations(3), rng=1)
    assert result.iterations == 3


def test_same_seed_same_result():
    def run(seed):
        return genetic_algorithm(
            make_problem(seed=seed), stop=max_iterations(10), rng=seed)
    a, b = run(42), run(42)
    assert (a.best_solution, a.best_value) == (b.best_solution, b.best_value)
    assert a.history == b.history
    assert run(1).history != run(2).history


def test_accepts_random_instance():
    result = genetic_algorithm(
        make_problem(), stop=max_iterations(2), rng=Random(3))
    assert result.iterations == 2


@pytest.mark.parametrize('direction', list(Direction))
def test_direction(direction):
    result = genetic_algorithm(
        make_problem(direction), stop=max_iterations(40), rng=0,
        population_size=20)
    values = [r['best'] for r in result.history]
    pick = min if direction is Direction.MINIMIZE else max
    assert result.best_value == pick(values)
    assert sum(result.best_solution) == result.best_value
    # the search moves towards the requested direction
    assert pick(values[0], result.best_value) == result.best_value
    assert result.best_value != values[0]
    for r in result.history:
        assert pick(r['best'], r['worst']) == r['best']
        assert sum(r['solution']) == r['best']


def test_result_contents():
    problem = make_problem()
    calls = 0

    def evaluate(genome):
        nonlocal calls
        calls += 1
        return sum(genome)

    problem = Problem(generate=problem.generate, evaluate=evaluate)
    result = genetic_algorithm(
        problem, stop=max_iterations(5), rng=0, population_size=8)
    assert result.iterations == 5
    assert len(result.history) == 6
    assert set(result.history[0]) == {
        'best', 'mean', 'worst', 'solution', 'best_so_far'}
    # one evaluation per genome per generation, selection reuses the scores
    assert result.metadata['evaluations'] == calls == 6 * 8
    assert result.metadata['termination'] == 'stop'
    state = result.metadata['state']
    assert isinstance(state, State) and state.iteration == 5
    assert state.evaluations == calls
    assert result.elapsed == state.elapsed >= 0


def test_stop_conditions():
    by_evals = genetic_algorithm(
        make_problem(), stop=max_evaluations(35), rng=0, population_size=10)
    # stop is checked per generation: may overshoot by one generation
    assert 35 <= by_evals.metadata['evaluations'] < 45
    assert by_evals.iterations == 3

    by_target = genetic_algorithm(
        make_problem(), stop=any_of(max_iterations(500), target_value(5)),
        rng=0, population_size=20)
    assert by_target.best_value <= 5
    assert by_target.iterations < 500

    # the State carries the problem's direction (it used to default to
    # MINIMIZE, so this stopped before the first generation)
    maximize = genetic_algorithm(
        make_problem(Direction.MAXIMIZE),
        stop=any_of(max_iterations(500), target_value(50)),
        rng=0, population_size=20)
    assert maximize.best_value >= 50
    assert 0 < maximize.iterations < 500

    immediate = genetic_algorithm(
        make_problem(), stop=max_iterations(0), rng=0)
    assert immediate.iterations == 0 and len(immediate.history) == 1


def test_only_feasible_genomes_are_evaluated():
    def feasible(genome):
        return sum(genome) <= 30

    def evaluate(genome):
        assert feasible(genome), genome
        return sum(genome)

    base = make_problem(Direction.MAXIMIZE)
    problem = Problem(generate=base.generate, evaluate=evaluate,
                      feasible=feasible, direction=Direction.MAXIMIZE)
    # crossover always breeds infeasible children, mutation cannot fix them
    result = genetic_algorithm(
        problem, stop=max_iterations(10), rng=0,
        crossover=lambda a, b, rng: ([9] * 6, [9] * 6),
        mutation=lambda g, rng: g[:])
    assert feasible(result.best_solution)


def test_repair_fixes_infeasible_children():
    def feasible(genome):
        return 0 not in genome

    problem = Problem(
        generate=lambda rng: [rng.randint(1, 9) for _ in range(4)],
        evaluate=sum, feasible=feasible)
    repaired = []

    def repair(genome):
        repaired.append(genome)
        return [g or 1 for g in genome]

    result = genetic_algorithm(
        problem, stop=max_iterations(5), rng=0, repair=repair,
        crossover=lambda a, b, rng: ([0] * 4, [0] * 4),
        mutation=lambda g, rng: g[:])
    assert repaired
    assert result.best_solution == [1, 1, 1, 1]


def test_unsatisfiable_problem_raises():
    problem = make_problem(feasible=lambda g: False)
    with pytest.raises(InfeasibleError):
        genetic_algorithm(problem, stop=max_iterations(1), max_tries=3)


@pytest.mark.parametrize('direction', list(Direction))
def test_selection_gets_oriented_scores(direction):
    seen = []

    def selection(population, scores, rng, k):
        assert k == 2
        seen.append((population, scores))
        return population[:k]

    problem = make_problem(direction)
    genetic_algorithm(problem, stop=max_iterations(2), rng=0,
                      population_size=6, selection=selection)
    for population, scores in seen:
        assert scores == [oriented(sum(g), direction) for g in population]


@pytest.mark.parametrize('direction', list(Direction))
def test_history_best_so_far_and_elitism(direction):
    result = genetic_algorithm(
        make_problem(direction), stop=max_iterations(15), rng=0)
    pick = min if direction is Direction.MINIMIZE else max
    so_far = [r['best_so_far'] for r in result.history]
    assert so_far[-1] == result.best_value
    for i, r in enumerate(result.history):
        assert r['best_so_far'] == pick(
            h['best'] for h in result.history[:i + 1])
        # elitism: the best so far is never lost from the population
        assert r['best'] == r['best_so_far']


def test_population_is_bred_not_resampled():
    generated = 0
    base = make_problem()

    def generate(rng):
        nonlocal generated
        generated += 1
        return base.generate(rng)

    genetic_algorithm(
        Problem(generate=generate, evaluate=sum), stop=max_iterations(5),
        rng=0, population_size=8)
    assert generated == 8  # only the initial population is random
