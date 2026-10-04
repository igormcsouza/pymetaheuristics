"""Genetic Algorithm: the reference heuristic (see docs/architecture.md).

One generation is an explicit pipeline of small steps::

    select -> crossover -> fill -> mutate -> feasibility -> evaluate -> record

Every genome that is evaluated (and so can become the best) is feasible.
"""
from random import Random
from statistics import fmean
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Union

from pymetaheuristics.core.direction import better, oriented
from pymetaheuristics.core.evaluation import counting
from pymetaheuristics.core.feasibility import InfeasibleError, reject
from pymetaheuristics.core.problem import Direction, Problem
from pymetaheuristics.core.result import OptimizationResult
from pymetaheuristics.core.termination import State, Stop
from pymetaheuristics.genetic_algorithm.steps.crossovers import (
    single_point_crossover)
from pymetaheuristics.genetic_algorithm.steps.multations import inter_mutation
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

    Each generation: ``selection`` picks parents, ``crossover`` breeds the
    first two, fresh genomes fill the population back to
    ``population_size``, every genome is mutated (``mutation`` is retried
    up to ``max_tries`` times for a feasible mutant, else the genome is kept
    as is), then any genome still infeasible (e.g. a crossover child) is
    passed through ``repair`` if given, and replaced by a fresh feasible
    genome if that does not fix it. ``operator_kwargs`` (e.g. ``k=5``) are
    forwarded to every operator, together with ``rng`` (and ``direction``
    for selection).

    ``stop`` is checked once per generation, so ``max_evaluations`` may be
    exceeded by up to one generation of evaluations.

    The result's ``history`` has one dict per generation, the initial
    population first (so ``len(history) == iterations + 1``)::

        {'best': float, 'mean': float, 'worst': float, 'solution': Genome}

    ``best``/``solution`` are that generation's best (not the best so far).
    ``metadata`` holds ``evaluations`` (objective calls) and
    ``termination``, the ``State`` that satisfied ``stop``.

    ``problem.generate`` is called without an rng; seed its random source
    yourself for fully reproducible runs.
    """
    rng = make_rng(rng)
    problem, evaluations = counting(problem)
    generate = reject(problem.generate, problem.feasible, max_tries)
    start = perf_counter()

    population = [generate() for _ in range(population_size)]
    values = [problem.evaluate(genome) for genome in population]
    record = _record(population, values, problem.direction)
    history = [record]
    best, best_value = record['solution'], record['best']
    generation = 0

    while True:
        state = State(
            generation, evaluations(), perf_counter() - start, best_value)
        if stop(state):
            break
        parents = _select(
            problem, population, values, selection, rng, operator_kwargs)
        offspring = _crossover(parents, crossover, rng, operator_kwargs)
        offspring += [
            generate() for _ in range(population_size - len(offspring))]
        offspring = _mutate(
            offspring, mutation, problem.feasible, max_tries, rng,
            operator_kwargs)
        population = _ensure_feasible(
            offspring, problem.feasible, repair, generate)
        values = [problem.evaluate(genome) for genome in population]
        record = _record(population, values, problem.direction)
        history.append(record)
        if better(record['best'], best_value, problem.direction):
            best, best_value = record['solution'], record['best']
        generation += 1

    return OptimizationResult(
        best, best_value, history=history, iterations=generation,
        elapsed=state.elapsed,
        metadata={'evaluations': state.evaluations, 'termination': state})


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


def _crossover(
    parents: Population, crossover: CrossOverFunction, rng: Random,
    kwargs: Dict[str, Any]
) -> Population:
    """Parents plus the children of the first two."""
    return [*parents, *crossover(*parents[:2], rng=rng, **kwargs)]


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


def _record(
    population: Population, values: List[float], direction: Direction
) -> Dict[str, Any]:
    """Per-generation statistics stored in the result history."""
    ranked = sorted(
        range(len(values)), key=lambda i: oriented(values[i], direction))
    return {
        'best': values[ranked[0]], 'mean': fmean(values),
        'worst': values[ranked[-1]], 'solution': population[ranked[0]]}
