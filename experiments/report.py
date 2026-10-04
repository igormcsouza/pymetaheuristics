"""Render experiments/results/summary.json into docs/experiments.md.

Rewrites only the block between the ``results`` markers; the hand-written
analysis around it is kept. Convergence plots (one PNG per benchmark
family) need the optional ``experiments`` group and are skipped without it:

    uv run --group experiments python -m experiments.report
"""
import json
from pathlib import Path

from experiments.runner import RESULTS, SUCCESS_GAP

DOCS = Path(__file__).parent.parent / 'docs'
START, END = '<!-- results:start -->', '<!-- results:end -->'


def table(summary):
    rows = [
        '| Benchmark | Heuristic | Gap mean | Gap std | Gap min | Gap max '
        '| Best value | Success | Evals | Time (ms) |',
        '|---|---|---|---|---|---|---|---|---|---|']
    rows += [
        '| %s | %s | %.4g | %.4g | %.4g | %.4g | %.4g | %.0f%% | %.0f | %.1f |'
        % (s['benchmark'], s['heuristic'], s['gap_mean'], s['gap_std'],
           s['gap_min'], s['gap_max'], s['value_best'],
           100 * s['success_rate'], s['evaluations_mean'],
           1000 * s['wall_time_mean'])
        for s in summary]
    return '\n'.join(rows)


def plot(summary, out=DOCS / 'img'):
    """One PNG per family: mean gap of the best-so-far vs evaluations."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed: skipping plots')
        return []
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for fam in dict.fromkeys(s['family'] for s in summary):
        rows = [s for s in summary if s['family'] == fam]
        benches = list(dict.fromkeys(s['benchmark'] for s in rows))
        fig, axes = plt.subplots(1, len(benches), squeeze=False,
                                 figsize=(5 * len(benches), 3.6))
        for ax, bench in zip(axes[0], benches):
            for s in rows:
                if s['benchmark'] == bench:
                    ax.plot(s['curve_evaluations'], s['curve_mean'],
                            marker='.', label=s['heuristic'])
            if all(v > 0 for s in rows for v in s['curve_mean']
                   if s['benchmark'] == bench):
                ax.set_yscale('log')  # labels minor ticks on short ranges
            else:
                ax.set_yscale('symlog', linthresh=SUCCESS_GAP)
            ax.set_title(bench)
            ax.set_xlabel('evaluations')
            ax.set_ylabel('mean gap of best so far')
            ax.grid(alpha=0.3)
        axes[0][0].legend()
        fig.tight_layout()
        path = out / ('experiments-%s.png' % fam)
        fig.savefig(path, dpi=100)
        plt.close(fig)
        paths.append(path)
    return paths


def render(summary, doc=DOCS / 'experiments.md'):
    text = doc.read_text()
    head, rest = text.split(START)
    tail = rest.split(END)[1]
    doc.write_text(
        '%s%s\n\n%s\n\n%s%s' % (head, START, table(summary), END, tail))


if __name__ == '__main__':
    summary = json.loads((RESULTS / 'summary.json').read_text())
    render(summary)
    plot(summary)
