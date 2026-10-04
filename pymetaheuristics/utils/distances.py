from math import sqrt
from typing import List


def euclidean_distance(a: List[float], b: List[float]):
    """Calculate a Euclidean distance between 2 tensors."""
    assert len(a) == len(b), "Length of tensor a has to be equal to tensor b."

    distances = list()
    for a_coord, b_coord in zip(a, b):
        distances.append((a_coord - b_coord) ** 2)

    return sqrt(sum(distances))


# ponytail: deprecated misspelled alias, remove in 0.3
euclidian_distance = euclidean_distance
