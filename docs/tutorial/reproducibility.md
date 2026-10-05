# Reproducibility

Same seed, same result.

Pass `rng=` (an `int` seed or a `random.Random`) to make a run
reproducible. The heuristic passes it to every operator. `problem.generate`
is called **without** an rng.

!!! warning
    Seed the random source of `generate` yourself, like the
    `seeded = Random(0)` used above, or runs will differ.

## Recap

- `rng=` seeds the heuristic and its operators.
- Seed `generate`'s own randomness too.

Next: [worked examples](../examples.md).
