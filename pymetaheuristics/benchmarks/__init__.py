"""Reusable benchmark suite: Problem instances with known best values.

Use `get(name)` / `all_benchmarks()`; see docs/benchmarks.md. Pass
an rng (or seed) to a heuristic for reproducible runs. Call the factories
(`knapsack`, `tsp`, `continuous`) for custom instances.
"""
from typing import Dict, List

from pymetaheuristics.benchmarks.benchmark import Benchmark, gap
from pymetaheuristics.benchmarks.continuous import (
    continuous, rastrigin_fn, sphere_fn)
from pymetaheuristics.benchmarks.knapsack import knapsack
from pymetaheuristics.benchmarks.tsp import tsp

_GRID9 = [[x, y] for x in range(3) for y in range(3)]
_RING8 = [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2], [1, 2], [0, 2], [0, 1]]

_LIST = [
    Benchmark(
        'knapsack-3', knapsack([60, 100, 120], [10, 20, 30], 50),
        220, [0, 1, 1], 'textbook instance; brute force'),
    Benchmark(
        'knapsack-10',
        knapsack([55, 10, 47, 5, 4, 50, 8, 61, 85, 87],
                 [95, 4, 60, 32, 23, 72, 80, 62, 65, 46], 269),
        295, [0, 1, 1, 1, 0, 0, 0, 1, 1, 1], 'brute force over 2^10'),
    Benchmark(
        'tsp-ring8', tsp(_RING8), 8.0, list(range(8)),
        'unit-step ring: every edge has length 1; brute force'),
    Benchmark(
        'tsp-grid9', tsp(_GRID9), 8 + 2 ** 0.5,
        [0, 1, 2, 5, 8, 7, 6, 3, 4],
        'unit 3x3 grid: 8 + sqrt(2); brute force'),
    Benchmark(
        'sphere-5', continuous(sphere_fn, 5, (-5.12, 5.12)),
        0.0, [0.0] * 5, 'analytic optimum at the origin'),
    Benchmark(
        'rastrigin-5', continuous(rastrigin_fn, 5, (-5.12, 5.12)),
        0.0, [0.0] * 5, 'analytic global optimum at the origin'),
]

BENCHMARKS: Dict[str, Benchmark] = {b.name: b for b in _LIST}


def get(name: str) -> Benchmark:
    return BENCHMARKS[name]


def all_benchmarks() -> List[Benchmark]:
    return list(BENCHMARKS.values())


__all__ = ['BENCHMARKS', 'Benchmark', 'all_benchmarks', 'continuous', 'gap',
           'get', 'knapsack', 'rastrigin_fn', 'sphere_fn', 'tsp']
