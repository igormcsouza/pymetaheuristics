# Benchmark suite

`pymetaheuristics.benchmarks` provides `Problem` instances with known best
values, independent of any heuristic. Use `get(name)`, `all_benchmarks()` or
`BENCHMARKS`; build custom instances with the factories `knapsack`, `tsp`,
`continuous` (build instances; randomness comes from the heuristic's `rng`).

| Name | Representation | Objective | Constraint | Known best |
|------|----------------|-----------|------------|-----------|
| `knapsack-3` | list of 0/1 per item | maximize value | weight <= capacity (50) | 220 (brute force) |
| `knapsack-10` | list of 0/1 per item | maximize value | weight <= capacity (269) | 295 (brute force) |
| `tsp-ring8` | permutation of 8 cities, closed tour | minimize length | valid permutation | 8.0 (brute force) |
| `tsp-grid9` | permutation of 3x3 grid points | minimize length | valid permutation | 8 + sqrt(2) (brute force) |
| `sphere-5` | 5 floats in [-5.12, 5.12] | minimize sum x^2 | within bounds | 0 (analytic) |
| `rastrigin-5` | 5 floats in [-5.12, 5.12] | minimize Rastrigin | within bounds | 0 (analytic) |

Each `Benchmark` holds `name`, `problem`, `known_best`,
`known_best_solution` (a feasible solution reaching it), `source` and
`criteria`.

## Evaluation criteria

Run a heuristic with a fixed budget and report `gap(benchmark, value)` for
its best feasible value: `|value - known_best| / max(|known_best|, 1)`, so 0
means optimal. Optionally also report evaluations needed to reach
`known_best` (e.g. via `target_value` termination). Compare heuristics on
the same benchmark and seed.
