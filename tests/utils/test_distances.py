from pymetaheuristics.utils.distances import euclidian_distance


def test_distances_euclidean():
    a = [1., 3.]
    b = [4., 7.]

    length = euclidian_distance(a, b)

    assert length == 5.
