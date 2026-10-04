import dataclasses

import pytest

from pymetaheuristics.core.result import OptimizationResult


def test_fields_and_defaults():
    r = OptimizationResult(best_solution=[1, 0], best_value=3.5)
    assert r.best_solution == [1, 0]
    assert r.best_value == 3.5
    assert r.history == []
    assert r.iterations == 0
    assert r.elapsed == 0.0
    assert r.metadata == {}


def test_full_construction():
    r = OptimizationResult(
        [1], 2.0, history=[{'best': 2.0}], iterations=1, elapsed=0.1,
        metadata={'termination': 'max_iterations'})
    assert r.history == [{'best': 2.0}]
    assert r.metadata['termination'] == 'max_iterations'


def test_frozen():
    r = OptimizationResult([1], 2.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.best_value = 0


def test_defaults_not_shared():
    a = OptimizationResult([1], 1.0)
    b = OptimizationResult([1], 1.0)
    a.history.append(1)
    assert b.history == []
