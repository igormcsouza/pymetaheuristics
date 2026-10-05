# Constraints and feasibility

`Problem.feasible` is the contract: **the heuristics only evaluate feasible
solutions.** Infeasible starts are drawn again, infeasible moves are retried
or dropped, and so the best solution returned is always feasible. Pick one
of three strategies, depending on how often `generate` and your operators
produce infeasible solutions.

=== "Reject"

    Draw again until the solution is feasible. This is what the
    heuristics already do with `generate`. The `reject` helper wraps any
    producer. If no feasible solution turns up within `max_tries`, it raises
    `InfeasibleError`.

    ```python
    from pymetaheuristics.core import reject

    feasible_packing = reject(knapsack.generate, knapsack.feasible,
                              max_tries=1000)
    assert knapsack.feasible(feasible_packing())
    ```

=== "Repair"

    Turn an infeasible solution into a feasible one. Rejection is
    wasteful when most random solutions are infeasible, and repair avoids that.
    You can give `genetic_algorithm` a `repair=` function for infeasible
    children, or wrap any producer with `repair(produce, fix)`:

    ```python
    from pymetaheuristics.core import repair


    def drop_heaviest(packing):
        packing = list(packing)
        while weight(packing) > CAPACITY:
            packed = [i for i, bit in enumerate(packing) if bit]
            packing[max(packed, key=lambda i: WEIGHTS[i])] = 0
        return packing


    repaired_generate = repair(knapsack.generate, drop_heaviest,
                               feasible=knapsack.feasible)
    assert knapsack.feasible(repaired_generate())
    ```

=== "Penalty"

    Allow infeasible solutions but score them worse.
    `penalty(problem, penalty_fn)` returns a new `Problem` where every solution
    is feasible and the objective is worsened by `penalty_fn(solution)` in the
    problem's direction. `penalty_fn` must return a value `>= 0` that is zero
    for feasible solutions.

    ```python
    from pymetaheuristics.core import penalty

    soft = penalty(knapsack, lambda s: 10 * max(0, weight(s) - CAPACITY))
    assert soft.evaluate([1, 1, 1, 1, 1]) == 390 - 10 * (90 - 50)
    ```

    !!! warning
        With a penalty, the best solution found may be infeasible under the
        original constraint. Check it with `knapsack.feasible`.

## Recap

- `feasible` is the contract: infeasible solutions are never evaluated.
- Reject when feasible solutions are common, repair when they are rare,
  penalty when the constraint can be soft.

Next: [decide when to stop](stopping.md).
