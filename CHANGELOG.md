# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/).

## [0.2.0]

### Added

- `Problem`, `Direction` and `OptimizationResult` core types, and the
  reference loop `core.run` with stopping criteria.
- `genetic_algorithm()` function on top of `core.run`.
- `simulated_annealing()` with pluggable neighborhoods and cooling.
- Feasibility handling through a single `feasible` predicate on `Problem`.
- Reproducible runs through the `rng` parameter.
- Benchmark problem suite, quality/performance experiments and extension
  examples.
- MkDocs documentation and a release workflow that attaches the sdist and
  wheel to the GitHub Release.

### Changed

- Requires Python 3.12+; tooling moved to uv, hatchling and ruff.
- Genetic operators no longer mutate their inputs.
- GA respects the objective `Direction` in selection and best tracking.
- Identifiers renamed to meaningful names.

### Deprecated

Removal planned for 0.3
([#50](https://github.com/igormcsouza/pymetaheuristics/issues/50)).

- `GeneticAlgorithm` class.
- `genetic_algorithm.steps.multations` shim (use `mutations`).
- `utils.distances.euclidian_distance` (use `euclidean_distance`).
- `inter_mutation(q=...)` (use `num_swaps=`).

### Fixed

- GA now breeds the whole population; selection is scale-invariant
  (weighted selection with positive fitness on Python 3.12).
- Exception handling fixes in the deprecated paths.

## [0.1.1]

Previous release; see the GitHub releases page.
