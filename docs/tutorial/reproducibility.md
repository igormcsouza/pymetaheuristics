# Reproducibility

Same seed, same result.

Pass `rng=` (an `int` seed or a `random.Random`) to make a run
reproducible. The heuristic passes it to every operator, including
`problem.generate(rng)`, so nothing else needs seeding.

## Recap

- `rng=` seeds the heuristic, its operators and `generate`.

Next: [worked examples](../examples.md).
