"""Genetic Algorithm: the reference heuristic (see docs/architecture.md).

One generation is an explicit pipeline of small steps::

    breed (select -> crossover, until full) -> mutate -> feasibility
    -> evaluate -> elitism -> record

Every genome that is evaluated (and so can become the best) is feasible.
"""
from functools import partial
from random import Random
from statistics import fmean
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pymetaheuristics.core.direction import better, oriented
from pymetaheuristics.core.feasibility import InfeasibleError, reject
from pymetaheuristics.core.problem import Direction, Problem
from pymetaheuristics.core.loop import run
from pymetaheuristics.core.result import OptimizationResult
from pymetaheuristics.core.termination import Stop
from pymetaheuristics.genetic_algorithm.steps.crossovers import (
    single_point_crossover)
from pymetaheuristics.genetic_algorithm.steps.mutations import inter_mutation
from pymetaheuristics.genetic_algorithm.steps.selections import (
    random_weighted_selection)
from pymetaheuristics.genetic_algorithm.types import (
    CrossOverFunction, Genome, MutationFunction, Population,
    SelectionFunction)
from pymetaheuristics.utils.rng import make_rng


def genetic_algorithm(
    problem: Problem, *, stop: Stop,
    rng: Optional[Union[Random, int]] = None,
    population_size: int = 10,
    selection: SelectionFunction = random_weighted_selection,
    crossover: CrossOverFunction = single_point_crossover,
    mutation: MutationFunction = inter_mutation,
    repair: Optional[Callable[[Genome], Genome]] = None,
    max_tries: int = 1000,
    **operator_kwargs: Any
) -> OptimizationResult:
    """Evolve a population of ``problem`` solutions until ``stop`` is met.

    Each generation breeds ``population_size`` children: ``selection``
    picks parents and ``crossover`` breeds the first two, repeated until
    the population is full. Every child is mutated (``mutation`` is retried
    up to ``max_tries`` times for a feasible mutant, else the child is kept
    as is), then any child still infeasible is passed through ``repair`` if
    given, and replaced by a fresh feasible genome if that does not fix it.
    After evaluation the best genome so far replaces the worst child if it
    is better (elitism), so it is never lost and never re-evaluated: each
    generation costs ``population_size`` evaluations (plus any made by a
    custom ``selection`` on genomes outside the population).
    ``operator_kwargs`` (e.g. ``k=5``) are
    forwarded to every operator, together with ``rng`` (and ``direction``
    for selection).

    ``stop`` is checked once per generation, so ``max_evaluations`` may be
    exceeded by up to one generation of evaluations.

    The result's ``history`` has one dict per generation, the initial
    population first (so ``len(history) == iterations + 1``)::

        {'best': float, 'mean': float, 'worst': float, 'solution': Genome,
         'best_so_far': float}

    ``best``/``solution`` are that generation's best; ``best_so_far`` is
    the best value seen up to and including that generation (the same
    measure as SA's float history). With elitism the two values coincide.
    ``metadata`` holds ``evaluations`` (objective calls), ``termination``
    ('stop') and ``state`` (the final ``State``), as for every heuristic
    built on ``core.run``.

    ``problem.generate`` is called with the run's rng.
    """
    rng = make_rng(rng)

    def init(problem):
        population = [generate() for _ in range(population_size)]
        return _evaluated(problem, population, None, None)

    def step(problem, carry):
        population, values, best, best_value = carry
        offspring = _breed(
            problem, population, values, selection, crossover,
            population_size, rng, operator_kwargs)
        offspring = _mutate(
            offspring, mutation, problem.feasible, max_tries, rng,
            operator_kwargs)
        population = _ensure_feasible(
            offspring, problem.feasible, repair, generate)
        return _evaluated(problem, population, best, best_value)

    generate = partial(
        reject(problem.generate, problem.feasible, max_tries), rng)
    return run(problem, stop=stop, init=init, step=step)


def _evaluated(
    problem: Problem, population: Population, best: Optional[Genome],
    best_value: Optional[float]
) -> Tuple[Any, Genome, float, Dict[str, Any]]:
    """Evaluate, apply elitism and record one generation, as a ``run``
    outcome; ``best`` is None for the initial population."""
    direction = problem.direction
    values = [problem.evaluate(genome) for genome in population]
    if best is not None:
        population, values = _keep_elite(
            population, values, best, best_value, direction)
    record = _record(population, values, direction)
    if best is None or better(record['best'], best_value, direction):
        best, best_value = record['solution'], record['best']
    record['best_so_far'] = best_value
    return ((population, values, best, best_value), record['solution'],
            record['best'], record)


def _select(
    problem: Problem, population: Population, values: List[float],
    selection: SelectionFunction, rng: Random, kwargs: Dict[str, Any]
) -> Population:
    """Run ``selection`` on cached fitness: members are not re-evaluated."""
    cache = {id(genome): value for genome, value in zip(population, values)}

    def fitness(genome: Genome) -> float:
        if id(genome) in cache:
            return cache[id(genome)]
        return problem.evaluate(genome)

    return selection(
        population, fitness, rng=rng, direction=problem.direction, **kwargs)


def _breed(
    problem: Problem, population: Population, values: List[float],
    selection: SelectionFunction, crossover: CrossOverFunction, size: int,
    rng: Random, kwargs: Dict[str, Any]
) -> Population:
    """``size`` children, each pair bred from freshly selected parents."""
    children: Population = []
    while len(children) < size:
        parents = _select(problem, population, values, selection, rng, kwargs)
        children += crossover(*parents[:2], rng=rng, **kwargs)
    return children[:size]


def _mutate(
    population: Population, mutation: MutationFunction,
    feasible: Callable[[Genome], bool], max_tries: int, rng: Random,
    kwargs: Dict[str, Any]
) -> Population:
    """Feasible mutant of each genome, or the genome itself if none found."""
    safe_mutation = reject(mutation, feasible, max_tries)

    def mutate(genome: Genome) -> Genome:
        try:
            return safe_mutation(genome, rng=rng, **kwargs)
        except InfeasibleError:
            return genome

    return [mutate(genome) for genome in population]


def _ensure_feasible(
    population: Population, feasible: Callable[[Genome], bool],
    repair: Optional[Callable[[Genome], Genome]],
    generate: Callable[[], Genome]
) -> Population:
    """Repair infeasible genomes, else replace them with fresh ones."""
    result = []
    for genome in population:
        if not feasible(genome) and repair is not None:
            genome = repair(genome)
        result.append(genome if feasible(genome) else generate())
    return result


def _keep_elite(
    population: Population, values: List[float], elite: Genome,
    elite_value: float, direction: Direction
) -> Tuple[Population, List[float]]:
    """Population and values with the worst replaced by ``elite`` if better."""
    worst = max(
        range(len(values)), key=lambda i: oriented(values[i], direction))
    if not better(elite_value, values[worst], direction):
        return population, values
    return ([*population[:worst], elite, *population[worst + 1:]],
            [*values[:worst], elite_value, *values[worst + 1:]])


def _record(
    population: Population, values: List[float], direction: Direction
) -> Dict[str, Any]:
    """Per-generation statistics stored in the result history."""
    ranked = sorted(
        range(len(values)), key=lambda i: oriented(values[i], direction))
    return {
        'best': values[ranked[0]], 'mean': fmean(values),
        'worst': values[ranked[-1]], 'solution': population[ranked[0]]}
