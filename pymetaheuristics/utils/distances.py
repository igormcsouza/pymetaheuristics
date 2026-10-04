import warnings
from math import dist
from typing import List


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Euclidean distance between 2 points (ValueError on length mismatch)."""
    return dist(a, b)


# ponytail: deprecated misspelled alias, remove in 0.3 (issue #50)
def euclidian_distance(a: List[float], b: List[float]) -> float:
    warnings.warn("euclidian_distance is deprecated, use euclidean_distance",
                  DeprecationWarning, stacklevel=2)
    return dist(a, b)
