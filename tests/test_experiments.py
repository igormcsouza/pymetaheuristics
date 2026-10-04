"""Smoke test of the experiment runner: tiny budget, 2 seeds."""
from experiments.report import table
from experiments.runner import aggregate, run, write
from pymetaheuristics.benchmarks import all_benchmarks

HEURISTICS = {'random_search', 'genetic_algorithm', 'simulated_annealing'}


def test_runner_smoke(tmp_path):
    runs = run(seeds=range(2), budget=40)
    assert len(runs) == len(all_benchmarks()) * len(HEURISTICS) * 2
    for r in runs:
        assert r['gap'] >= 0 and r['termination'] == 'stop'
        assert 40 <= r['evaluations'] <= 50  # GA may finish its generation
        assert len(r['curve']) == 20
        assert all(a >= b for a, b in zip(r['curve'], r['curve'][1:]))

    summary = aggregate(runs)
    assert {s['heuristic'] for s in summary} == HEURISTICS
    assert all(s['seeds'] == 2 and 0 <= s['success_rate'] <= 1
               for s in summary)
    assert all(s['gap_min'] <= s['gap_mean'] <= s['gap_max']
               for s in summary)

    write(runs, summary, tmp_path)
    assert (tmp_path / 'runs.csv').read_text().count('\n') == len(runs) + 1
    assert (tmp_path / 'summary.json').exists()
    assert table(summary).count('\n') == len(summary) + 1


def test_runs_are_reproducible():
    tsp = [b for b in all_benchmarks() if b.name == 'tsp-grid9']
    first, second = (run(tsp, seeds=[3], budget=30) for _ in range(2))
    assert [r['curve'] for r in first] == [r['curve'] for r in second]
