import pytest

from pymetaheuristics.utils.distances import euclidean_distance


def test_distances_euclidean():
    assert euclidean_distance([1., 3.], [4., 7.]) == 5.


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        euclidean_distance([1.], [1., 2.])
