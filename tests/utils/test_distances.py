from pymetaheuristics.utils.distances import euclidean_distance


def test_distances_euclidean():
    a = [1., 3.]
    b = [4., 7.]

    length = euclidean_distance(a, b)

    assert length == 5.
