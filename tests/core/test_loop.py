import pytest

from pymetaheuristics.core import (
    Direction, Problem, State, any_of, make_rng, max_evaluations,
    max_iterations, run, target_value)


def problem(direction=Direction.MINIMIZE):
    return Problem(generate=lambda rng: 0, evaluate=lambda x: x,
                   direction=direction)


def walker(delta, record=None):
    """init/step pair: x moves by ``delta`` per step, one evaluation each."""
    def init(problem):
        x = problem.generate(None)
        return x, x, problem.evaluate(x), record

    def step(problem, x):
        x += delta
        return x, x, problem.evaluate(x), record

    return init, step


@pytest.mark.parametrize('direction, delta, best', [
    (Direction.MINIMIZE, -1, -5), (Direction.MAXIMIZE, 1, 5),
    (Direction.MINIMIZE, 1, 0), (Direction.MAXIMIZE, -1, 0)])
def test_best_so_far_respects_direction(direction, delta, best):
    init, step = walker(delta)
    r = run(problem(direction), stop=max_iterations(5), init=init, step=step)
    assert r.best_value == r.best_solution == best
    assert r.iterations == 5 and len(r.history) == 6
    assert r.history[-1] == best  # None record -> best so far


def test_evaluations_metadata_and_final_state():
    init, step = walker(-1)
    r = run(problem(), stop=max_evaluations(4), init=init, step=step)
    state = r.metadata['state']
    assert r.metadata['evaluations'] == 4 == state.evaluations
    assert r.metadata['termination'] == 'stop'
    assert isinstance(state, State) and state.iteration == r.iterations == 3
    assert state.best_value == r.best_value
    assert r.elapsed == state.elapsed


def test_target_value_reads_direction_from_state():
    init, step = walker(1)
    r = run(problem(Direction.MAXIMIZE),
            stop=any_of(max_iterations(100), target_value(3)),
            init=init, step=step)
    assert r.best_value == 3 and r.iterations == 3
    assert r.metadata['state'].direction is Direction.MAXIMIZE


def test_records_and_extras():
    init, step = walker(-1, record={'tag': 1})
    r = run(problem(), stop=max_iterations(2), init=init, step=step,
            extras=lambda carry: {'last': carry})
    assert r.history == [{'tag': 1}] * 3
    assert r.metadata['last'] == -2


def test_stop_before_any_step():
    init, step = walker(-1)
    r = run(problem(), stop=max_iterations(0), init=init, step=step)
    assert r.iterations == 0 and r.history == [0]


def test_make_rng_exported():
    assert make_rng(3).random() == make_rng(3).random()
