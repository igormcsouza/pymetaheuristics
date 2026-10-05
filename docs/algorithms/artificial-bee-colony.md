# Artificial Bee Colony: theory and implementation

This page explains where `artificial_bee_colony()` comes from, the mathematics
of its three phases, how the Python code maps onto them, where it differs from
the original algorithm, and how well it performs today. Every claim about
behaviour was checked against `pymetaheuristics/artificial_bee_colony/` and
`neighborhoods.py`.

## Background

Artificial Bee Colony (ABC) was proposed by Karaboga[^karaboga] as a model of
the foraging of honey bees, and analysed and compared with other population
methods by Karaboga and Basturk[^kb2007][^kb2008]. A colony has three kinds of
bees:

- **employed bees**, each tied to one *food source* (a candidate solution),
  which look for a better source close to it;
- **onlooker bees**, which watch the employed bees and choose which sources
  deserve more search, favouring the good ones;
- **scout bees**, which abandon a source that stopped improving and look for a
  new one at random.

The first two phases exploit; the third explores. ABC is greedy and
population-based, with one parameter (`limit`) that controls how long a source
may stagnate. There is no temperature and no crossover.

## One iteration

Let \(N\) be `colony_size`, the number of food sources
\(x_1,\dots,x_N\), with oriented values \(v_i = \text{oriented}(f(x_i))\)
(lower is better for both directions, see `core.oriented`) and a *trial
counter* \(t_i\) per source, initially 0. The initial colony is \(N\) feasible
solutions from `problem.generate(rng)`.

### 1. Employed bees

Each source \(i\) gets one candidate \(x_i'\) from its neighbourhood and keeps
it only if it is **strictly** better:

\[
x_i \leftarrow
\begin{cases}
x_i' & v(x_i') < v_i \quad (t_i \leftarrow 0)\\
x_i & \text{otherwise} \quad (t_i \leftarrow t_i + 1)
\end{cases}
\]

In code (`colony.py`) the same helper serves the employed and onlooker phases:

```python
value = problem.evaluate(candidate)
if better(value, values[i], direction):
    sources[i], values[i], trials[i] = candidate, value, 0
else:
    trials[i] += 1
```

If no feasible neighbour is found after `max_neighbor_tries` redraws, the try
counts as a failure and costs no evaluation.

### 2. Onlooker bees

\(N\) onlookers each pick a source and try a neighbour of it exactly as above.
Two indices \(a, b\) are drawn uniformly (with replacement) and the better of
the two sources is chosen (ties go to \(a\)): a **binary tournament**. For
distinct values, the source of rank \(r\) (1 = best) among \(N\) is chosen
with probability

\[
P(r) = \frac{2(N - r) + 1}{N^2}
\]

so the best source is chosen \(2N - 1\) times more often than the worst, and
the probabilities sum to 1. Good sources get more search, as in the original,
but the pressure depends only on rank, not on the scale of the objective.

### 3. Scouts

A source whose counter exceeds `limit` (\(t_i > L\)) is abandoned and replaced
by a fresh feasible `problem.generate(rng)` solution, which is evaluated, with
\(t_i = 0\). The best solution so far is kept by `core.run`, so abandoning a
source never loses the best.

## Differences from the original

| | Karaboga's ABC | This implementation |
|---|---|---|
| Candidate | \(v_{ij} = x_{ij} + \varphi_{ij}(x_{ij} - x_{kj})\), \(\varphi \sim U(-1,1)\), \(k \ne i\), \(j\) a random coordinate | any `neighbor(solution, rng)` |
| Onlooker choice | roulette, \(p_i = \text{fit}_i / \sum_j \text{fit}_j\) with \(\text{fit}_i = 1/(1+f_i)\) if \(f_i \ge 0\), else \(1 + \lvert f_i\rvert\) | binary tournament (rank-based) |
| Scout | at most one source per cycle, the one past `limit`; new point uniform in the box | every source past `limit`, new point from `problem.generate` |
| Domain | continuous | any representation with a neighbourhood |

The original update moves a source along the difference to another source,
which is specific to real vectors. Replacing it with the `neighbor` contract is
what lets the same loop solve TSP (`swap_neighbor`, `two_opt_neighbor`),
knapsack (`bit_flip_neighbor`) and continuous problems (`gaussian_neighbor`).
The price is that with a *fixed* step the search no longer adapts its step
size to how spread the colony is (in the original, \(\lvert x_{ij} - x_{kj}
\rvert\) shrinks as the colony converges). The roulette formula for
\(\text{fit}\) also needs a sign convention for the objective; the tournament
avoids it. Neither choice is tuned; both are replaceable.

## Parameters

| Parameter | Default | Effect |
|---|---|---|
| `colony_size` \(N\) | 20 | number of sources, and of employed and of onlooker bees. A bigger colony explores more but each source gets fewer tries from a fixed budget. |
| `limit` \(L\) | 50 | failed tries before a source is abandoned. Small \(L\) explores more, large \(L\) exploits. For \(D\)-dimensional continuous problems it is commonly set proportional to \(N \cdot D\); see the comparative study of Karaboga and Akay[^ka]. |
| `neighbor` | `swap_neighbor` | the move; the step size lives here |
| `max_neighbor_tries` | 100 | redraws for a feasible neighbour |
| `max_start_tries` | 1000 | redraws for a feasible initial or scout solution |

!!! tip "Defaults are generic"
    As with SA, set `colony_size`, `limit` and the neighbour's step for your
    problem. The experiments use \(N = 10\) and \(L = 20\), not tuned per
    instance.

## Cost model

Per iteration, the employed phase makes at most \(N\) evaluations, the
onlooker phase at most \(N\) more, and each scout one more. The initial colony
costs \(N\). After \(T\) iterations:

\[
E \;\le\; N + 2NT + S
\]

where \(S\) is the number of scouts that fired. Fewer evaluations are made when
infeasible neighbours exhaust their redraws. `stop` is checked once per
iteration, so `max_evaluations` can be overshot by up to one iteration (about
\(2N\) evaluations).

The result's `history` is the best value so far, once per iteration with the
initial colony first, so it is monotone and has `iterations + 1` entries.

## Convergence

ABC has no convergence guarantee here. Greedy acceptance means a source never
gets worse, scouts add random restarts, and with an unbounded run on a finite
problem the restarts eventually sample the space, but that says nothing about
the speed. Treat it as a heuristic.

## Current scores

Protocol (see [Experiments](../experiments.md)): 2000 evaluations, 20 seeds,
\(N = 10\), \(L = 20\). Mean *gap* to the known optimum (0 is optimal).

| Benchmark | ABC gap mean | ABC success | SA gap | GA gap | Random search gap |
|---|---|---|---|---|---|
| knapsack-3 | 0 | 100% | 0 | 0 | 0 |
| knapsack-10 | 0.00017 | 95% | 0.00017 | 0 | 0 |
| tsp-ring8 | 0 | 100% | 0 | 0 | 0.097 |
| tsp-grid9 | 0 | 100% | 0 | 0 | 0.047 |
| sphere-5 | 0.70 | 20% | 4.4e-3 | 8.5e-5 | 2.0 |
| rastrigin-5 | 12.6 | 0% | 41.7 | 13.4 | 22.0 |

Reading the table:

- ABC solves both TSP instances on every seed, and is the fastest heuristic
  per run (about 9-17 ms for 2000 evaluations).
- On `sphere-5` it beats random search but is far behind the GA and SA, and
  the variance is high (best seed about \(7 \cdot 10^{-5}\), worst about
  3.4). Changing `limit` from 20 to 1000 changes nothing, so abandoned
  sources are not the cause. Halving the colony to 5 sources lowers the mean
  gap to about 0.11, which suggests the 2000 evaluations are spread too
  thinly over 10 sources for a fixed step of 0.1. This was observed, not
  proven.
- On `rastrigin-5` it has the best mean gap by a small margin (12.6 vs 13.4
  for the GA), plausibly because scouts keep restarting in new basins, but its
  best seed (about 3.1) is not better than SA's best (about 3.0).

Reproduce with `uv run python -m experiments.runner`.

## References

[^karaboga]: D. Karaboga, "An idea based on honey bee swarm for numerical
    optimization", Technical Report TR06, Erciyes University, Engineering
    Faculty, Computer Engineering Department, 2005.
[^kb2007]: D. Karaboga and B. Basturk, "A powerful and efficient algorithm for
    numerical function optimization: artificial bee colony (ABC) algorithm",
    *Journal of Global Optimization* 39(3), 2007, pp. 459-471.
[^kb2008]: D. Karaboga and B. Basturk, "On the performance of artificial bee
    colony (ABC) algorithm", *Applied Soft Computing* 8(1), 2008, pp. 687-697.
[^ka]: D. Karaboga and B. Akay, "A comparative study of artificial bee colony
    algorithm", *Applied Mathematics and Computation* 214(1), 2009, pp. 108-132.
