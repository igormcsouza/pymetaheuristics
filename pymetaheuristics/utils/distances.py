from math import dist
from typing import List


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Euclidean distance between 2 points (ValueError on length mismatch)."""
    return dist(a, b)

