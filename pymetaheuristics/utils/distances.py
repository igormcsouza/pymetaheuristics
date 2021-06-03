from math import sqrt
from typing import List


def euclidian_distance(a: List[float], b: List[float]):
    """Calculate a Euclidian distance between 2 tensors."""
    assert len(a) == len(b), "Length of tensor A has to be equal to tensor B."

    distances = list()
    for a_axis, b_axis in zip(a, b):
        distances.append((a_axis - b_axis) ** 2)

    return sqrt(sum(distances))
