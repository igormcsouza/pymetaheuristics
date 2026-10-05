# Artificial Bee Colony: theory and implementation

Artificial Bee Colony (ABC) was proposed by Karaboga (2005) after the foraging
behaviour of honey bees. A colony of `colony_size` *food sources* (solutions)
is improved by three kinds of bees, all implemented in
`pymetaheuristics/artificial_bee_colony/colony.py`.

## One iteration

1. **Employed bees**: every source tries one neighbour and keeps it only if it
   is strictly better (greedy acceptance, honouring `Problem.direction`). A
   failed try increments the source's *trial counter*; a success resets it.
2. **Onlooker bees**: `colony_size` times, two sources are drawn at random and
   the better one is chosen (binary tournament), then it gets the same
   try-a-neighbour treatment. Good sources are therefore exploited more. The
   classic fitness-proportional roulette is replaced by the tournament so that
   negative or unbounded values need no scaling.
3. **Scouts**: a source whose counter exceeds `limit` is abandoned and
   replaced by a fresh `problem.generate()` solution.

## Usage

```python
from pymetaheuristics.artificial_bee_colony import (
    artificial_bee_colony, gaussian_neighbor)
from pymetaheuristics.benchmarks.continuous import continuous, sphere_fn
from pymetaheuristics.core import max_iterations

problem = continuous(sphere_fn, 3, (-5, 5), rng=0)
result = artificial_bee_colony(
    problem, stop=max_iterations(100), rng=0, neighbor=gaussian_neighbor)
print(result.best_value)
```

The `neighbor` is any `neighbor(solution, rng)` function, so permutation
(`swap_neighbor`, `two_opt_neighbor`), binary (`bit_flip_neighbor`) and
continuous (`gaussian_neighbor`) problems all work. Infeasible neighbours are
redrawn up to `max_neighbor_tries` times, else the try counts as a failure.

One iteration costs about `2 * colony_size` evaluations (more when scouts
fire), so `max_evaluations` may be overshot by up to one iteration.
