import pytest

from pymetaheuristics.utils.distances import (
    euclidean_distance, euclidian_distance)


def test_distances_euclidean():
    assert euclidean_distance([1., 3.], [4., 7.]) == 5.


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        euclidean_distance([1.], [1., 2.])


def test_misspelled_alias_warns():
    with pytest.warns(DeprecationWarning):
        assert euclidian_distance([0., 0.], [3., 4.]) == 5.
