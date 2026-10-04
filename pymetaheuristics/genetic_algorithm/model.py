import warnings
from typing import List, Optional, Tuple, Union
from random import Random
from time import time

from pymetaheuristics.core.problem import Direction, Problem
from pymetaheuristics.core.termination import max_iterations
from pymetaheuristics.genetic_algorithm.algorithm import genetic_algorithm
from pymetaheuristics.genetic_algorithm.types import (
    ConstraintFunction, CrossOverFunction, FitnessFunction,
    GeneticAlgorithmHistory, Genome, GenomeGeneratorFunction, MutationFunction,
    SelectionFunction)
from pymetaheuristics.genetic_algorithm.steps.selections import (
    random_weighted_selection)
from pymetaheuristics.genetic_algorithm.steps.crossovers import (
    single_point_crossover)
from pymetaheuristics.genetic_algorithm.steps.mutations import inter_mutation
from pymetaheuristics.genetic_algorithm.exceptions import LoadHistoryException


class GeneticAlgorithm():
    """Deprecated: use ``genetic_algorithm(problem, stop=..., ...)`` from
    ``pymetaheuristics.genetic_algorithm.algorithm``. This class is a thin
    backwards-compatible wrapper around it and will be removed later.

    ## A Genetic Representation of a Real World Problem

    Genetic Algorithm tries to mimic what nature does for natural selection to
    solve real world problems. A problem may be represent as an Genetic Problem
    if can be model with:

        A Genome (Genetic Representation of a Solution)
        A Fitness Function (Score to Rank the Genome)
        A Selection Method (Ways to choose between Genomes)
        A Crossover Method (Ways to shuffle Genomes)
        A Mutation Method (Ways to change small pieces of a Genome)

    On the Step Module you can find some functions to help on those matters,
    and may fit perfectly on you problem, or maybe one might need to implement
    their own algorithm to each of this steps.

    ** Note: If you think your problem has not a function to help on those
    steps, feel free to open a issue so your code, or someelse's code may
    become part of the package too.

    Pass ``direction=Direction.MAXIMIZE`` to maximize the fitness function
    (default is MINIMIZE); no need to negate it (See Knapsack model on test
    folder). To look for a specific result, minimize the difference.
    """

    def __init__(
        self, fitness_function: FitnessFunction,
        genome_generator: GenomeGeneratorFunction,
        constraints: Optional[List[ConstraintFunction]] = None,
        max_tries: int = 1000,
        direction: Direction = Direction.MINIMIZE
    ):
        warnings.warn(
            "GeneticAlgorithm is deprecated; use genetic_algorithm(problem, "
            "stop=...) instead. It will be removed in 0.3 (see issue #50).",
            DeprecationWarning, stacklevel=2)
        self.direction = direction
        self.fitness_function = fitness_function
        self.genome_generator = genome_generator
        # thin adapter: constraints are just a Problem.feasible predicate
        self.constraints = list(constraints or [])
        self.max_tries = max_tries
        self.history: GeneticAlgorithmHistory = {}

    def load_history(self, history: GeneticAlgorithmHistory):
        """check if the given history is on the right pattern."""
        if not hasattr(history, "keys") or len(history.keys()) == 0:
            raise LoadHistoryException(
                "The given history of has not the correct pattern. "
                "It must be a non-empty mapping.")

        top = ("args", "runs", "best", "elapsed")
        args = ("epochs", "pop_size", "selection", "crossover", "mutation",
                "verbose", "kwargs")
        for key, entry in history.items():
            if not hasattr(entry, "keys") or any(k not in entry for k in top):
                raise LoadHistoryException(
                    "The given history of %s has not the correct pattern. "
                    "Expected keys %s." % (key, top))
            if (not hasattr(entry["args"], "keys")
                    or any(k not in entry["args"] for k in args)):
                raise LoadHistoryException(
                    "The given history of %s has not the correct pattern. "
                    "Expected args %s." % (key, args))

        # If everything is ok, update the history
        self.history.update(history)

    def add_constraint(self, constraint: ConstraintFunction):
        """Genetic Contraint for a Gene."""
        self.constraints.append(constraint)

    def _check_constraints(self, genome: Genome):
        for constraint_it in self.constraints:
            if not constraint_it(genome):
                return False

        return True

    def train(
        self,
        epochs: int,
        pop_size: int,
        selection: SelectionFunction = random_weighted_selection,
        crossover: CrossOverFunction = single_point_crossover,
        mutation: MutationFunction = inter_mutation,
        verbose: bool = False,
        rng: Optional[Union[Random, int]] = None,
        **kwargs
    ) -> Tuple[Genome, float]:
        """Run ``genetic_algorithm`` for ``epochs`` generations (deprecated).

        Below is the Hyperparameters one can change to get better results.
        Just a quick note, all the problems have their specific parameters,
        it maybe mean the default will not work for one's case.

        Parameters:
        :epochs: Number of evolutions algorithm will loop over
        :pop_size: Number of Genomes (Genetic Representation of a solution)
        :selection: Selection funtion to be used (See Steps Module)
        :crossover: Crossover funtion to be used (See Steps Module)
        :multation: Mutation funtion to be used (See Steps Module)
        :rng: A random.Random or an int seed, for reproducible runs.

        Optional Parameters:
        Depending on the function one choses, it might come with optional
        parameters that can be set as parameters on this function. The code
        will automatically deal with it.
        """
        problem = Problem(
            generate=self.genome_generator, evaluate=self.fitness_function,
            feasible=self._check_constraints, direction=self.direction)
        result = genetic_algorithm(
            problem, stop=max_iterations(epochs), rng=rng,
            population_size=pop_size, selection=selection,
            crossover=crossover, mutation=mutation, max_tries=self.max_tries,
            **kwargs)
        # history[0] is the initial population; runs are one per epoch
        runs = [(r['solution'], r['best']) for r in result.history[1:]]
        if verbose:
            for i, (genome, fitness) in enumerate(runs):
                print("Epoch %i got fitness %.2f" % (i, fitness), genome)

        self.history[time()] = {
            "runs": runs,
            "best": (result.best_solution, result.best_value),
            "elapsed": result.elapsed,
            "args": {
                "epochs": epochs, "pop_size": pop_size,
                "selection": selection.__name__,
                "crossover": crossover.__name__,
                "mutation": mutation.__name__,
                "verbose": verbose, "kwargs": kwargs
            }
        }
        return result.best_solution, result.best_value
