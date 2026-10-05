# Genetic Algorithm: theory and implementation

This page explains where `genetic_algorithm()` comes from, the mathematics
behind each operator, how the Python code maps onto it, and how well it
performs today. Every claim about behaviour was checked against the code in
`pymetaheuristics/genetic_algorithm/`.

## Background

Genetic algorithms were introduced by Holland[^holland] and popularised by
Goldberg[^goldberg]. A population of \(N\) candidate solutions (genomes)
evolves for \(G\) generations. In each generation fitter genomes are more
likely to become parents, parents are recombined (*crossover*), and children
are randomly perturbed (*mutation*). The expected effect is that building
blocks of good solutions are kept and combined.

Holland's schema theorem gives the intuition. For a schema \(H\) (a pattern
of fixed genes) with fitness \(f(H)\), defining length \(\delta(H)\), order
\(o(H)\), mean population fitness \(\bar f\), crossover probability \(p_c\),
mutation probability \(p_m\) and genome length \(\ell\):

\[
\mathbb{E}\,[m(H, t+1)] \;\ge\; m(H, t)\,\frac{f(H)}{\bar f}
\Bigl[\,1 - p_c\,\frac{\delta(H)}{\ell - 1} - p_m\,o(H)\Bigr]
\]

Short, low-order, above-average schemata receive exponentially more samples.
The theorem is a statement about proportional selection, so it is only a
guide here: the library scales fitness (below), and it makes no claim of
global convergence.

## The generation loop

One generation is the pipeline in `algorithm.py`:

```text
breed (select -> crossover, until full) -> mutate -> feasibility
  -> evaluate -> elitism -> record
```

| Step | Function | Source of the idea |
|---|---|---|
| Selection | `random_weighted_selection` | fitness-proportionate (roulette wheel)[^goldberg] |
| Crossover | `single_point_crossover` | Holland[^holland] |
| Permutation crossover | `pmx_single_point` | PMX, Goldberg & Lingle[^pmx] |
| Mutation | `inter_mutation` | adjacent-gene swap |
| Elitism | `_keep_elite` | De Jong[^dejong] |

## Selection: scaled roulette wheel

Plain roulette selection picks genome \(i\) with probability
\(p_i = f_i / \sum_j f_j\). That depends on the *scale* of the objective: on
values between 0.001 and 0.002 the probabilities are almost uniform, and it
is undefined for negative values. The library works in oriented values
\(v_i\) (lower is better for both directions, see `core.oriented`) and builds
weights by linear scaling:

\[
w_i = (v_{\max} - v_i) + \lambda\,(v_{\max} - v_{\min}), \qquad
p_i = \frac{w_i}{\sum_j w_j}, \qquad \lambda = 0.1
\]

Properties, all visible in the code of `selections.py`:

- The best genome gets the largest weight, the worst gets
  \(\lambda\,(v_{\max}-v_{\min}) > 0\), so every genome stays selectable.
- Multiplying all values by a constant, or adding one, leaves \(p_i\)
  unchanged. Selection pressure does not depend on the objective's scale.
- If all values are equal the spread is 0 and selection is uniform.

For example, values \([10, 20, 40]\) minimized give weights
\([33, 23, 3]\) and probabilities \(\approx [0.56, 0.39, 0.05]\).

`k = 2` parents are drawn *with replacement* (`random.choices`), so a genome
can mate with itself. They are returned as copies.

## Crossover

### Single point (bit lists, float lists)

A cut \(c\) is drawn uniformly from \(\{1,\dots,\ell-1\}\) and tails are
swapped: \(\text{child}_1 = p_1[:c] \Vert p_2[c:]\),
\(\text{child}_2 = p_2[:c] \Vert p_1[c:]\). Each child is a valid genome for
unconstrained representations. For permutations it is not, because genes
repeat, which is why the next operator exists.

### PMX (permutations)

Partially mapped crossover[^pmx] copies a segment from one parent and
resolves the resulting duplicates through the mapping between the two
segments. `pmx_single_point` is the special case where the segment is the
prefix \([0, c)\). For each \(i < c\) it swaps, inside a copy of \(p_1\), the
gene at \(i\) with the position holding \(p_2[i]\):

```python
child1 = parent1[:]
for i in range(cut_point):
    partner_index = child1.index(parent2[i])
    child1[partner_index] = child1[i]
    child1[i] = parent2[i]
```

Each step is a transposition, so \(\text{child}_1\) is always a permutation.
After the loop, \(\text{child}_1[:c] = p_2[:c]\), and the remaining genes keep
the positions they had in \(p_1\) wherever that does not collide. The same is
done with the parents exchanged for \(\text{child}_2\). The cost is
\(O(c\,\ell)\) because of `list.index`.

## Mutation: bounded adjacent swaps

`inter_mutation` repeats up to `num_swaps = 2` times: draw an index, stop with
probability \(1 - 0.75\), otherwise swap the gene with its left neighbour. The
number of swaps \(S\) satisfies \(P(S \ge k) = 0.75^k\), so

\[
\mathbb{E}[S] = \sum_{k=1}^{2} 0.75^k = 1.3125 .
\]

Adjacent swaps preserve permutations, but they are a small move. For TSP the
benchmarks use `two_opt_neighbor` (Croes[^croes]) through a one-line adapter,
and for knapsack `bit_flip_neighbor`.

## Elitism and feasibility

After evaluation the best genome seen so far replaces the worst child if it is
strictly better (`_keep_elite`). This guarantees the best-so-far is monotone
non-increasing in oriented value and is never lost, at no extra evaluation
because its value is cached. De Jong[^dejong] showed elitism improves the
on-line performance of simple GAs.

Constraints are handled by *reject / repair / replace*: a mutant is redrawn
up to `max_tries` times, then `repair` is applied, and an unrepairable child
is replaced by a fresh feasible genome. Hence every evaluated genome is
feasible.

## Cost model

One generation costs exactly \(N\) evaluations (population size), so a budget
of \(B\) evaluations gives \(\lceil B/N \rceil\) generations. `stop` is
checked once per generation, so the budget can be overshot by at most
\(N - 1\) evaluations.

## Current scores

Protocol (see [Experiments](../experiments.md)): 2000 evaluations, 20 seeds,
population 10, mean *gap* to the known optimum (0 is optimal). Random search
is the baseline.

| Benchmark | GA gap mean | GA success | Random search gap | Simulated annealing gap |
|---|---|---|---|---|
| knapsack-3 | 0 | 100% | 0 | 0 |
| knapsack-10 | 0 | 100% | 0 | 0.0005 |
| tsp-ring8 | 0 | 100% | 0.097 | 0 |
| tsp-grid9 | 0 | 100% | 0.047 | 0 |
| sphere-5 | 8.0e-5 | 100% | 2.0 | 3.8e-3 |
| rastrigin-5 | 17.6 | 0% | 22.0 | 41.6 |

Reading the table:

- On TSP and on the small knapsacks the GA reaches the optimum in every seed.
  The knapsack instances have at most 1024 packings, fewer than the budget,
  so they do not separate the GA from random search.
- On `sphere-5` it is the best of the three heuristics, because every child is
  mutated and elitism keeps the best.
- On `rastrigin-5` it beats random search but never reaches the optimum.
  With only 10 genomes, fitness-weighted selection and elitism the population
  collapses into one basin within a few generations (*premature
  convergence*). A larger population, a larger mutation step or a
  diversity-preserving selection are the usual remedies.

Reproduce with `uv run python -m experiments.runner`.

## References

[^holland]: J. H. Holland, *Adaptation in Natural and Artificial Systems*,
    University of Michigan Press, 1975.
[^goldberg]: D. E. Goldberg, *Genetic Algorithms in Search, Optimization and
    Machine Learning*, Addison-Wesley, 1989.
[^pmx]: D. E. Goldberg and R. Lingle, "Alleles, loci and the traveling
    salesman problem", *Proc. First International Conference on Genetic
    Algorithms*, 1985, pp. 154-159.
[^dejong]: K. A. De Jong, *An Analysis of the Behavior of a Class of Genetic
    Adaptive Systems*, PhD thesis, University of Michigan, 1975.
[^croes]: G. A. Croes, "A method for solving traveling-salesman problems",
    *Operations Research* 6(6), 1958, pp. 791-812.
