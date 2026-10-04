# Experiments

Evidence that the heuristics actually optimize, not just run: every
heuristic on every benchmark of the [suite](benchmarks.md), with the same
objective budget, over many seeds.

```sh
uv run python -m experiments.runner                      # results only
uv run --group experiments python -m experiments.report  # table + plots
```

`experiments/` lives outside the installable package. `runner.py` needs
only the library and writes `experiments/results/runs.csv` (one row per
run) and `summary.json` (aggregates and mean convergence curves);
`report.py` rewrites the table below and, if matplotlib (optional
`experiments` dependency group) is installed, the PNG plots.

## Protocol

- **Budget:** `max_evaluations(2000)` for every run, so heuristics are
  compared at equal objective cost. The GA checks `stop` once per
  generation and may overshoot by one generation; with a population of 10
  it lands exactly on 2000 here.
- **Seeds:** 20 per (benchmark, heuristic), `0..19`. The seed drives both
  the heuristic's `rng` and a private copy of the benchmark's `generate`
  stream (benchmarks seed theirs once at build time, so runs would
  otherwise share one stream).
- **Metrics:** final best value, `gap(benchmark, value)` (0 = optimal;
  absolute for the continuous optima at 0), success rate (final gap
  <= 1e-3), evaluation count, wall time, and the gap of the best-so-far
  at 20 evenly spaced evaluation counts (convergence curve). The curve is
  recorded by wrapping `evaluate`, so it is in evaluations for every
  heuristic regardless of the shape of `result.history`.

## Configurations

| Family | SA move / GA mutation | GA crossover | SA T0 |
|---|---|---|---|
| knapsack | `bit_flip_neighbor` | `single_point_crossover` | 50 |
| tsp | `two_opt_neighbor` | `pmx_single_point` | 1 |
| continuous | `gaussian_neighbor` (local, sigma 0.1, one coordinate, clipped) | `single_point_crossover` | 10 |

- **Random search** (local baseline in `experiments/runner.py`): fresh
  feasible solutions from `problem.generate` until the budget is spent.
- **Genetic algorithm:** library defaults (population 10, weighted
  selection of 2 parents); the SA neighborhood is adapted as a mutation.
- **Simulated annealing:** geometric cooling from T0 to T0/1000 over the
  budget (one evaluation per iteration). T0 is set to the order of a
  typical worsening move for the family, not tuned per instance.

The library has no float operators, so `gaussian_neighbor` lives in
`experiments/` rather than expanding the package.

## Results

<!-- results:start -->

| Benchmark | Heuristic | Gap mean | Gap std | Gap min | Gap max | Best value | Success | Evals | Time (ms) |
|---|---|---|---|---|---|---|---|---|---|
| knapsack-3 | random_search | 0 | 0 | 0 | 0 | 220 | 100% | 2000 | 8.3 |
| knapsack-3 | genetic_algorithm | 0 | 0 | 0 | 0 | 220 | 100% | 2000 | 11.0 |
| knapsack-3 | simulated_annealing | 0 | 0 | 0 | 0 | 220 | 100% | 2000 | 4.9 |
| knapsack-10 | random_search | 0 | 0 | 0 | 0 | 295 | 100% | 2000 | 18.9 |
| knapsack-10 | genetic_algorithm | 0 | 0 | 0 | 0 | 295 | 100% | 2000 | 22.3 |
| knapsack-10 | simulated_annealing | 0.0005085 | 0.001617 | 0 | 0.00678 | 295 | 90% | 2000 | 6.4 |
| tsp-ring8 | random_search | 0.09723 | 0.08795 | 0 | 0.1768 | 8 | 45% | 2000 | 13.2 |
| tsp-ring8 | genetic_algorithm | 0.07071 | 0.0866 | 0 | 0.1768 | 8 | 60% | 2000 | 17.7 |
| tsp-ring8 | simulated_annealing | 0 | 0 | 0 | 0 | 8 | 100% | 2000 | 12.1 |
| tsp-grid9 | random_search | 0.04663 | 0.05558 | 0 | 0.1502 | 9.414 | 55% | 2000 | 13.7 |
| tsp-grid9 | genetic_algorithm | 0.03309 | 0.04455 | 0 | 0.1313 | 9.414 | 60% | 2000 | 18.5 |
| tsp-grid9 | simulated_annealing | 0 | 0 | 0 | 0 | 9.414 | 100% | 2000 | 12.8 |
| sphere-5 | random_search | 2 | 0.9487 | 0.5834 | 4.102 | 0.5834 | 0% | 2000 | 4.6 |
| sphere-5 | genetic_algorithm | 1.623 | 0.6422 | 0.6351 | 3.117 | 0.6351 | 0% | 2000 | 8.7 |
| sphere-5 | simulated_annealing | 0.003816 | 0.001668 | 0.001932 | 0.007247 | 0.001932 | 0% | 2000 | 5.1 |
| rastrigin-5 | random_search | 22.04 | 3.987 | 11.93 | 27.22 | 11.93 | 0% | 2000 | 5.2 |
| rastrigin-5 | genetic_algorithm | 20.04 | 4.865 | 13.4 | 32 | 13.4 | 0% | 2000 | 9.2 |
| rastrigin-5 | simulated_annealing | 41.56 | 25.68 | 2.991 | 108.5 | 2.991 | 0% | 2000 | 5.8 |

<!-- results:end -->

![knapsack convergence](img/experiments-knapsack.png)
![tsp convergence](img/experiments-tsp.png)
![continuous convergence](img/experiments-continuous.png)

## Findings

- **TSP: SA clearly wins.** 2-opt SA reaches the optimum in all 20 seeds
  on both instances within ~400 evaluations; GA and random search are
  still 3-10% off on average after 2000.
- **Sphere: SA is the only one that converges**, to a gap of ~4e-3. That
  is still above the 1e-3 success threshold: with a fixed sigma of 0.1
  the last steps are too coarse, and the early high-temperature phase
  spends half the budget wandering (its curve starts worst).
- **Rastrigin: SA performs poorly, worse than random search** (mean gap
  ~42 vs ~22, std ~26). Single-coordinate steps of 0.1 rarely cross
  between basins that are ~1 apart, so once the temperature drops the run
  freezes in whichever local minimum it wandered into. Its best seed
  (gap ~3) is the best overall, so it is high-variance rather than
  useless. No heuristic gets near the optimum with this budget.
- **GA is barely better than random search everywhere.** This follows
  from the library's generation pipeline: only 2 parents + 2 children are
  kept and the other 6 of 10 genomes are fresh random ones each
  generation, so most of the budget is random sampling. A larger
  population makes that worse, not better.
- **Knapsack does not discriminate.** `knapsack-3` has 8 packings and
  `knapsack-10` 1024, fewer than the budget, so random search is 100%
  successful. SA is sensitive to T0 here: a first run with T0 = 10 (below
  the item values, 4-120) trapped SA in local optima (40% / 20% success);
  T0 = 50 gives 100% / 90%. Harder instances are needed to say more.
- **Time:** every run takes a few to ~25 ms; the GA is the slowest per
  evaluation (selection, crossover and feasibility bookkeeping), SA the
  fastest.

## Known inconsistency

`metadata['termination']` is the `State` that met `stop` in the GA but the
string `'stop'` in SA. The runner normalizes both to `'stop'` (see the
`termination` column of `runs.csv`); library semantics are unchanged here.
