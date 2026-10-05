# Simulated Annealing: theory and implementation

This page explains where `simulated_annealing()` comes from, the mathematics
of its acceptance rule and cooling schedule, how the Python code maps onto
them, and how well it performs today. Every claim about behaviour was checked
against `pymetaheuristics/simulated_annealing/` and `neighborhoods.py`.

## Background

Simulated annealing (SA) was proposed independently by Kirkpatrick, Gelatt and
Vecchi[^kgv] and by Černý[^cerny]. It borrows the Metropolis algorithm[^metropolis]
from statistical physics: a system in thermal equilibrium at temperature
\(T\) occupies a state with energy \(E\) with probability proportional to the
Boltzmann factor \(e^{-E/T}\). Slowly lowering \(T\) (annealing) drives the
system towards low-energy states. Treating the objective as the energy turns
this into an optimizer that can escape local minima, because at high \(T\) it
accepts worse solutions.

## The Metropolis acceptance rule

At each iteration a neighbour \(x'\) of the current solution \(x\) is drawn.
With \(\Delta = v(x') - v(x)\) measured in *oriented* value (lower is better
for both directions, see `core.oriented`), the move is accepted with
probability

\[
P(\text{accept}) =
\begin{cases}
1 & \Delta \le 0 \\[2pt]
e^{-\Delta / T} & \Delta > 0
\end{cases}
\]

In code (`annealing.py`):

```python
delta = oriented(value, direction) - oriented(current_value, direction)
if delta <= 0 or (temperature > 0
                  and rng.random() < math.exp(-delta / temperature)):
    current, current_value = candidate, value
```

Because \(\Delta\) is oriented, the same code minimizes and maximizes, and
`temperature > 0` avoids dividing by zero once a linear schedule reaches 0.

For a *fixed* \(T\), a Metropolis chain over a connected neighbourhood with a
symmetric proposal has the Boltzmann distribution \(\pi_T(x) \propto
e^{-v(x)/T}\) as its stationary distribution. As \(T \to 0\), \(\pi_T\)
concentrates on the global optima. That is the theoretical reason annealing
works. Note that `swap_neighbor`, `two_opt_neighbor` and `bit_flip_neighbor`
are symmetric proposals, so this applies to the built-in neighbourhoods.

## Cooling schedules

| Schedule | Formula | Function |
|---|---|---|
| Geometric | \(T_{k+1} = \alpha\,T_k\), so \(T_k = T_0\,\alpha^k\) | `geometric_cooling(alpha)` |
| Linear | \(T_{k+1} = \max(T_k - s,\ 0)\) | `linear_cooling(step)` |

A schedule is any function `temperature -> temperature`, applied once per
iteration.

!!! info "What is and is not guaranteed"
    Hajek[^hajek] proved that SA converges in probability to a global optimum
    if \(T_k = c / \ln(k + 2)\) with \(c\) at least the depth of the deepest
    non-global local minimum. That schedule is far too slow to be practical,
    and the library does not ship it. Geometric cooling, the practical choice
    popularised by Kirkpatrick et al., carries **no convergence guarantee**: it
    is a heuristic that trades the guarantee for speed. A custom
    `cooling` function can implement the logarithmic schedule if you need it.

### Choosing \(\alpha\) from a budget

To cool from \(T_0\) to \(T_0 / R\) over \(B\) iterations (one evaluation
each), solve \(\alpha^B = 1/R\):

\[
\alpha = R^{-1/B}
\]

The experiments use \(R = 1000\) and \(B = 2000\), giving
\(\alpha = 0.001^{1/2000} \approx 0.99655\):

```python
geometric_cooling(0.001 ** (1 / budget))
```

### Choosing \(T_0\)

The acceptance probability of a worsening move of size \(d\) is
\(e^{-d/T_0}\) at the start. Set \(T_0\) to about the size of a typical
worsening move, so early iterations accept a substantial fraction of them. The
experiments use 50 for knapsack (item values 4 to 120), 1 for TSP (edge
lengths around 1) and 10 for the continuous functions, none tuned per
instance. The defaults of the function (`initial_temperature=100.0`,
`geometric_cooling(0.95)`) are generic, so set them for your problem.

## Neighbourhoods

| Function | Move | Representation |
|---|---|---|
| `bit_flip_neighbor` | flip one bit | 0/1 lists |
| `swap_neighbor` | swap two positions | permutations |
| `two_opt_neighbor` | reverse a segment | permutations |

`two_opt_neighbor` is the 2-opt move of Croes[^croes]: reversing the segment
\(i..j\) replaces two edges of a tour by two others, which is the natural
local move for TSP because it changes only two edges.

## Feasibility

`Problem.feasible` is enforced by redrawing: an infeasible neighbour is
redrawn up to `max_neighbor_tries` times (default 100). If none is feasible the
iteration keeps the current solution, so the chain stays inside the feasible
set. As a consequence, near a very constrained boundary the effective proposal
is not exactly symmetric.

## Cost model

Each iteration costs at most one evaluation (zero if no feasible neighbour
was found). The temperature is cooled every iteration, even when the move
was rejected.

## Current scores

Protocol (see [Experiments](../experiments.md)): 2000 evaluations, 20 seeds,
geometric cooling to \(T_0/1000\). Mean *gap* to the known optimum (0 is
optimal).

| Benchmark | SA gap mean | SA success | Random search gap | Genetic algorithm gap |
|---|---|---|---|---|
| knapsack-3 | 0 | 100% | 0 | 0 |
| knapsack-10 | 0.0005 | 90% | 0 | 0 |
| tsp-ring8 | 0 | 100% | 0.097 | 0 |
| tsp-grid9 | 0 | 100% | 0.047 | 0 |
| sphere-5 | 3.8e-3 | 0% | 2.0 | 8.0e-5 |
| rastrigin-5 | 41.6 | 0% | 22.0 | 17.6 |

Reading the table:

- SA solves both TSP instances in every seed, with the 2-opt move.
- On `knapsack-10` it misses the optimum in 2 of 20 seeds. It is sensitive to
  \(T_0\): with \(T_0 = 10\), below the item values, it got stuck in local
  optima in 60-80% of the seeds.
- On `sphere-5` it gets close but not within \(10^{-3}\): the fixed Gaussian
  step of 0.1 is too coarse at the end, and the hot first half wanders.
- On `rastrigin-5` the mean is worse than random search, because SA freezes in
  a local minimum, but the variance is high and its best seed (gap about 3)
  is the best of all heuristics.

Reproduce with `uv run python -m experiments.runner`.

## References

[^kgv]: S. Kirkpatrick, C. D. Gelatt and M. P. Vecchi, "Optimization by
    simulated annealing", *Science* 220(4598), 1983, pp. 671-680.
[^cerny]: V. Černý, "Thermodynamical approach to the traveling salesman
    problem: an efficient simulation algorithm", *Journal of Optimization
    Theory and Applications* 45, 1985, pp. 41-51.
[^metropolis]: N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller
    and E. Teller, "Equation of state calculations by fast computing
    machines", *Journal of Chemical Physics* 21(6), 1953, pp. 1087-1092.
[^hajek]: B. Hajek, "Cooling schedules for optimal annealing", *Mathematics of
    Operations Research* 13(2), 1988, pp. 311-329.
[^croes]: G. A. Croes, "A method for solving traveling-salesman problems",
    *Operations Research* 6(6), 1958, pp. 791-812.
