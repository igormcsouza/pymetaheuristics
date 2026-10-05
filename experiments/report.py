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


def plot_overview(summary, out=DOCS / 'img'):
    """Two PNGs for the landing pages: share of random search's final gap
    each heuristic closes (0 = no better than the baseline, 100 = optimum;
    worse than random is shown as 0) and mean time per run.
    Benchmarks where random search already finds the optimum (gap 0) are
    left out of the first chart: the ratio is undefined."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    out.mkdir(parents=True, exist_ok=True)
    by = {(s['benchmark'], s['heuristic']): s for s in summary}
    benches = list(dict.fromkeys(s['benchmark'] for s in summary))
    heuristics = [h for h in dict.fromkeys(s['heuristic'] for s in summary)
                  if h != 'random_search']
    hard = [b for b in benches if by[b, 'random_search']['gap_mean'] > 0]

    def bars(ax, names, value, title, ylabel):
        width = 0.8 / len(heuristics)
        for i, h in enumerate(heuristics):
            ax.bar([x + i * width for x in range(len(names))],
                   [value(n, h) for n in names], width, label=h)
        ax.set_xticks([x + 0.4 - width / 2 for x in range(len(names))])
        ax.set_xticklabels(names, fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)

    paths = []
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars(ax, hard, lambda b, h: 100 * max(
             1 - by[b, h]['gap_mean'] / by[b, 'random_search']['gap_mean'],
             0),
         'Gap to the optimum closed, vs random search',
         '% of random-search gap closed')
    ax.set_ylim(0, 135)
    ax.set_yticks(range(0, 101, 20))
    ax.legend(fontsize=7, loc='upper center', ncol=3)
    paths.append(out / 'overview-gap.png')
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars(ax, benches, lambda b, h: 1000 * by[b, h]['wall_time_mean'],
         'Time per run (2000 evaluations)', 'mean wall time (ms)')
    ax.legend(fontsize=8)
    paths.append(out / 'overview-time.png')
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=100)
    plt.close(fig)
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
    plot_overview(summary)
