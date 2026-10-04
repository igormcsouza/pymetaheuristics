"""Quality/performance experiments: every heuristic on every benchmark.

Each run gets the same objective budget (``max_evaluations``) so heuristics
are comparable. No plotting here: this module needs only the library.

    uv run python -m experiments.runner   # writes experiments/results/
"""
import csv
import json
import types
from dataclasses import replace
from functools import partial
from pathlib import Path
from random import Random
from statistics import fmean, pstdev
from time import perf_counter

from pymetaheuristics.benchmarks import all_benchmarks, gap
from pymetaheuristics.core import (
    OptimizationResult, State, better, counting, max_evaluations, oriented,
    reject)
from pymetaheuristics.genetic_algorithm import genetic_algorithm
from pymetaheuristics.genetic_algorithm.steps.crossovers import (
    pmx_single_point, single_point_crossover)
from pymetaheuristics.simulated_annealing import (
    bit_flip_neighbor, geometric_cooling, simulated_annealing,
    two_opt_neighbor)

RESULTS = Path(__file__).parent / 'results'
SUCCESS_GAP = 1e-3  # a run "succeeds" when its final gap is at most this
CHECKPOINTS = 20    # convergence curve points per run
BOUNDS = (-5.12, 5.12)  # box of the continuous benchmarks


def family(benchmark_name):
    prefix = benchmark_name.split('-')[0]
    return 'continuous' if prefix in ('sphere', 'rastrigin') else prefix


# --- local operators (missing from the library for floats) -----------------

def gaussian_neighbor(solution, rng, sigma=0.1, bounds=BOUNDS):
    """Float move: add N(0, sigma) to one coordinate, clipped to bounds."""
    new = list(solution)
    i = rng.randrange(len(new))
    new[i] = min(max(new[i] + rng.gauss(0, sigma), bounds[0]), bounds[1])
    return new


def as_mutation(move):
    """Adapt an SA ``neighbor(solution, rng)`` to the GA mutation shape."""
    return lambda genome, rng=None, **_: move(genome, rng)


def random_search(problem, *, stop, rng=None):
    """Baseline: evaluate fresh feasible solutions until ``stop``."""
    problem, evaluations = counting(problem)
    generate = reject(problem.generate, problem.feasible)
    start = perf_counter()
    best = generate()
    best_value = problem.evaluate(best)
    history, iteration = [best_value], 0
    while not stop(State(iteration, evaluations(), perf_counter() - start,
                         best_value)):
        candidate = generate()
        value = problem.evaluate(candidate)
        if better(value, best_value, problem.direction):
            best, best_value = candidate, value
        iteration += 1
        history.append(best_value)
    return OptimizationResult(
        best, best_value, history=history, iterations=iteration,
        elapsed=perf_counter() - start,
        metadata={'evaluations': evaluations(), 'termination': 'stop'})


# --- configurations ---------------------------------------------------------

# per family: (move/mutation, GA crossover, SA initial temperature ~ size of
# a typical worsening move)
_FAMILY = {
    'knapsack': (bit_flip_neighbor, single_point_crossover, 50.0),
    'tsp': (two_opt_neighbor, pmx_single_point, 1.0),
    'continuous': (gaussian_neighbor, single_point_crossover, 10.0),
}


def configs(family_name, budget):
    """Heuristic name -> ``f(problem, stop, rng)`` for one problem family.

    SA cools geometrically from T0 to T0/1000 over the budget (one
    evaluation per iteration). GA uses the library default population (10)
    and selection.
    """
    move, crossover, t0 = _FAMILY[family_name]
    return {
        'random_search': random_search,
        'genetic_algorithm': partial(
            genetic_algorithm, population_size=10, crossover=crossover,
            mutation=as_mutation(move)),
        'simulated_annealing': partial(
            simulated_annealing, neighbor=move, initial_temperature=t0,
            cooling=geometric_cooling(0.001 ** (1 / budget))),
    }


# --- running ----------------------------------------------------------------

def _seeded(problem, seed):
    """Copy of ``problem`` whose ``generate`` draws from ``Random(seed)``.

    Benchmarks seed their generator once at build time, so all runs would
    share (and advance) one stream; this gives each run its own.
    """
    # ponytail: swaps Random cells of the closure; works for the suite's
    # factories, a generate without a closed-over Random is left as is
    gen = problem.generate
    cells = tuple(
        types.CellType(Random(seed))
        if isinstance(c.cell_contents, Random) else c
        for c in gen.__closure__ or ())
    return replace(problem, generate=types.FunctionType(
        gen.__code__, gen.__globals__, gen.__name__, gen.__defaults__,
        cells))


def run_one(benchmark, name, heuristic, seed, budget):
    """One run; the curve is the gap of the best-so-far at CHECKPOINTS
    evenly spaced evaluation counts up to ``budget``.

    Assumes the heuristic only evaluates feasible solutions (true for the
    GA, SA and random search here).
    """
    problem = _seeded(benchmark.problem, seed)
    trace = []  # best value so far after each evaluation

    def evaluate(solution):
        value = problem.evaluate(solution)
        trace.append(value if not trace or better(
            value, trace[-1], problem.direction) else trace[-1])
        return value

    start = perf_counter()
    result = heuristic(replace(problem, evaluate=evaluate),
                       stop=max_evaluations(budget), rng=seed)
    wall_time = perf_counter() - start
    # GA stores the State that met `stop`, SA a string: normalize both
    termination = result.metadata.get('termination')
    if isinstance(termination, State):
        termination = 'stop'
    final_gap = gap(benchmark, result.best_value)
    steps = [budget * (i + 1) // CHECKPOINTS for i in range(CHECKPOINTS)]
    return {
        'benchmark': benchmark.name, 'family': family(benchmark.name),
        'heuristic': name, 'seed': seed, 'best_value': result.best_value,
        'gap': final_gap, 'success': final_gap <= SUCCESS_GAP,
        'evaluations': len(trace), 'iterations': result.iterations,
        'wall_time': wall_time, 'termination': termination,
        'curve': [gap(benchmark, trace[min(s, len(trace)) - 1])
                  for s in steps],
        'curve_evaluations': steps,
    }


def run(benchmarks=None, seeds=range(20), budget=2000):
    """Every (benchmark, heuristic config, seed) run, as a list of dicts."""
    runs = []
    for benchmark in benchmarks or all_benchmarks():
        fam = family(benchmark.name)
        for name, heuristic in configs(fam, budget).items():
            runs += [run_one(benchmark, name, heuristic, seed, budget)
                     for seed in seeds]
    return runs


def aggregate(runs, benchmarks=None):
    """One summary dict per (benchmark, heuristic), in run order."""
    directions = {b.name: b.problem.direction
                  for b in benchmarks or all_benchmarks()}
    groups = {}
    for r in runs:
        groups.setdefault((r['benchmark'], r['heuristic']), []).append(r)
    summary = []
    for (bench, name), rs in groups.items():
        values = [r['best_value'] for r in rs]
        gaps = [r['gap'] for r in rs]
        ranked = sorted(values, key=lambda v: oriented(v, directions[bench]))
        summary.append({
            'benchmark': bench, 'family': rs[0]['family'], 'heuristic': name,
            'seeds': len(rs),
            'value_mean': fmean(values), 'value_std': pstdev(values),
            'value_best': ranked[0], 'value_worst': ranked[-1],
            'gap_mean': fmean(gaps), 'gap_std': pstdev(gaps),
            'gap_min': min(gaps), 'gap_max': max(gaps),
            'success_rate': fmean(r['success'] for r in rs),
            'evaluations_mean': fmean(r['evaluations'] for r in rs),
            'wall_time_mean': fmean(r['wall_time'] for r in rs),
            'curve_evaluations': rs[0]['curve_evaluations'],
            'curve_mean': [fmean(c) for c in zip(*(r['curve'] for r in rs))],
        })
    return summary


def write(runs, summary, out=RESULTS):
    """runs.csv (one row per run, no curve) and summary.json."""
    out.mkdir(parents=True, exist_ok=True)
    fields = [k for k in runs[0] if not k.startswith('curve')]
    with open(out / 'runs.csv', 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(runs)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')


if __name__ == '__main__':
    runs = run()
    summary = aggregate(runs)
    write(runs, summary)
    for s in summary:
        print('%-12s %-20s gap %.4f +- %.4f  success %3.0f%%' % (
            s['benchmark'], s['heuristic'], s['gap_mean'], s['gap_std'],
            100 * s['success_rate']))
