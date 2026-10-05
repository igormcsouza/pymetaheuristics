# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/).

## [0.3.0]

### Added

- `artificial_bee_colony()`.

### Changed

- `Problem.generate` takes the run's `rng`.
- GA operators take `rng`: `mutation(genome, rng, ...)`,
  `crossover(parent1, parent2, rng)`, `selection(population, scores, rng, k)`
  with oriented scores. Knobs are bound with `functools.partial`;
  `**operator_kwargs` is gone.
- `inter_mutation(genome, rng, num_swaps=2, probability=0.75)`.

### Removed

- `GeneticAlgorithm` class, `GeneticAlgorithmHistory`, `LoadHistoryException`.
- `genetic_algorithm.steps.multations` (use `mutations`).
- `utils.distances.euclidian_distance` (use `euclidean_distance`).
- `inter_mutation(q=...)` (use `num_swaps=`).
- Neighborhood re-exports from the simulated annealing module (use
  `pymetaheuristics.neighborhoods`).

See [Upgrading from 0.2](docs/release-notes.md).

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
