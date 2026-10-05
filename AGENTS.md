# AGENTS.md

Guide for AI agents (and humans) changing this repository. Read it before
you start a feature or a bug fix. The goal is a consistent, small, modular
package.

## Philosophy

pymetaheuristics is **functional and modular**. Keep it that way.

- A heuristic is a plain function `heuristic(problem, *, stop, rng=None, ...)`
  that returns an `OptimizationResult` (see `core/heuristic.py`). No base
  classes, no registries, no framework.
- Every moving part is a plain function the user can replace: operators
  (selection, crossover, mutation), neighborhoods (`neighbor(solution, rng)`),
  cooling schedules, repair functions, stop conditions, `Problem` callbacks.
  Algorithm knobs are keyword arguments with defaults, bound with
  `functools.partial`. No `**kwargs` swallowing, no hidden coupling between
  operators and the algorithm.
- Algorithms are built on the shared `core.run` loop (init/step/stop,
  counting, best-so-far, result). New algorithms reuse it and the existing
  helpers (`reject`, `repair`, `penalty`, `neighborhoods`, `make_rng`)
  before writing anything new.
- Randomness always comes from the run's `rng` (`Random`), including
  `Problem.generate(rng)`. Same seed, same result.
- A user must be able to swap any of our functions for their own without
  touching the package. If a change makes that harder, the design is wrong.

## Before you write code

1. **Understand first.** Read the code the change touches and trace the real
   flow end to end. Grep every caller of what you are about to change.
2. **Is the change minimal and at the right level?** Fix the root cause once,
   where all callers route through, not the symptom in one path.
3. **Smell check.** If the change feels like a bandaid (an adapter, a special
   case, a flag, a `**kwargs` escape hatch, a copy of existing logic, or
   hundreds of lines that a better design would make unnecessary), **stop and
   discuss it with the human first.** Propose the design fix and wait for a
   decision. Do not ship the bandaid and do not silently redesign.
4. Prefer deleting code to adding it. Reuse before writing. No abstraction
   without a second user, no config for a value that never changes.
5. Breaking changes are a decision for the human. Raise them, do not slip them in.

## Consistency rules

- **Existing tests must keep passing.** A change that breaks them either has
  a bug or is a deliberate contract change; in the second case say so, get
  the human's agreement, and update tests, docs and the changelog together.
- New behavior comes with tests. Project coverage is 100%; keep it. Test
  determinism with a seed, both `Direction`s, feasibility and stop conditions
  where relevant.
- Match the surrounding code: naming, docstring style, comment density.
  Ruff with line length 79 (`uv run ruff check .`); pre-commit runs it.
- Operators never modify their inputs and return new objects.
- Type-annotate public functions. Keep modules small.

## Documentation is part of the change

Every change that a user can see must leave documentation in `docs/`, so
people understand what it does and why it helps.

- Update the relevant tutorial, `extending.md`, `reference/` page and
  `release-notes.md`/`CHANGELOG.md` (the latter for anything user-facing or
  breaking, with before/after snippets and replacements).
- Python code blocks in the docs and README are executed by
  `tests/test_docs.py`: they must run. Use ```` ```text ```` for snippets
  that cannot.
- Add new pages to `mkdocs.yml` nav. The build must pass:
  `uv run --group docs mkdocs build --strict`.
- Keep README and `docs/index.md` consistent with each other.

### New metaheuristics

A new metaheuristic ships with all of:

1. the package `pymetaheuristics/<name>/` (a function over `core.run`,
   exported from its `__init__.py`);
2. tests under `tests/<name>/`;
3. a user page in `docs/algorithms/` and an API page in `docs/reference/`;
4. **an article in `docs/advance/`** with the mathematical explanation of
   the implementation: the model, the update/acceptance rules with their
   equations, parameters and their effect, complexity and evaluation cost
   per iteration, and where this implementation deliberately differs from
   the literature. Cite the original and relevant papers (authors, title,
   venue, year);
5. registration in `experiments/runner.py` (with the same per-family
   moves), regenerated results (`uv run python -m experiments.runner`,
   then `uv run --group experiments python -m experiments.report`), and an
   honest entry in `docs/experiments.md`. Report results as measured; do
   not tune per instance to look good.

## Commands

```sh
uv sync                                      # dev tools
uv run ruff check .                          # lint
sh scripts/test.sh                           # pytest with coverage, incl. doc snippets
uv run --group docs mkdocs build --strict    # docs
uv run python -m experiments.runner          # experiments (results only)
```

Run lint, tests and the docs build before opening a PR.

## Workflow

- Work on a branch, open a PR, never push to `main`. Keep one concern per PR.
- The PR description says what changed and why, and mentions any breaking
  change or follow-up.
- Releases are cut by the maintainer (see `docs/releasing.md`); do not tag
  or publish.
