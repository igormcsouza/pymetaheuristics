from typing import List, Optional, Tuple, Union
from random import Random
from time import time

from pymetaheuristics.core.direction import better, oriented
from pymetaheuristics.core.problem import Direction
from pymetaheuristics.genetic_algorithm.types import (
    ConstraintFunction, CrossOverFunction, FitnessFunction,
    GeneticAlgorithmHistory, Genome, GenomeGeneratorFunction, MutationFunction,
    SelectionFunction)
from pymetaheuristics.genetic_algorithm.steps.selections import (
    random_weighted_selection)
from pymetaheuristics.genetic_algorithm.steps.crossovers import (
    single_point_crossover)
from pymetaheuristics.genetic_algorithm.steps.multations import inter_mutation
from pymetaheuristics.genetic_algorithm.exceptions import LoadHistoryException
from pymetaheuristics.core.feasibility import InfeasibleError, reject
from pymetaheuristics.utils.rng import make_rng


class GeneticAlgorithm():
    """## A Genetic Representation of a Real World Problem

    Genetic Algorithm tries to mimic what nature does for natural selection to
    solve real world problems. A problem may be represent as an Genetic Problem
    if can be model with:

        A Genome (Genetic Representation of a Solution)
        A Fitness Function (Score to Rank the Genome)
        A Selection Method (Ways to choose between Genomes)
        A Crossover Method (Ways to shuffle Genomes)
        A Multation Method (Ways to change small pieces of a Genome)

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
        self.direction = direction
        self.fitness_function = fitness_function
        self.genome_generator = genome_generator
        # thin adapter: constraints are just a Problem.feasible predicate
        self.constraints = list(constraints or [])
        self.max_tries = max_tries
        self.history: GeneticAlgorithmHistory = {}

    def load_history(self, history: GeneticAlgorithmHistory):
        """check if the given history is on the right pattern."""
        # are there keys on history?
        try:
            assert history is not None
            assert len(history.keys()) > 0
        except AssertionError as ae:
            raise LoadHistoryException(
                "The given history of has not the correct pattern. %s" % ae)
        except Exception as e:
            raise LoadHistoryException(
                "An unexpected error occured. %s" % e)

        for keys in history.keys():
            # Get list of arguments
            try:
                parameters = history[keys].keys()
                # are there args?
                assert "args" in parameters
                arguments = history[keys]["args"].keys()  # type: ignore
                # are there runs?
                assert "runs" in parameters
                # is there best?
                assert "best" in parameters
                # is there elapsed?
                assert "elapsed" in parameters

                assert "epochs" in arguments
                assert "pop_size" in arguments
                assert "selection" in arguments
                assert "crossover" in arguments
                assert "mutation" in arguments
                assert "verbose" in arguments
                assert "kwargs" in arguments
            except AssertionError as ae:
                raise LoadHistoryException(
                    "The given history of %s has not the correct pattern. %s"
                    % (keys, ae))
            except Exception as e:
                raise LoadHistoryException(
                    "An unexpected error occured. %s" % e)

        # If everything is ok, update the history
        self.history.update(history)

    def _pop_generator(self, pop_size: int) -> List[Genome]:
        """Generate a population of genomes."""
        generate = reject(
            self.genome_generator, self._check_constraints, self.max_tries)
        return [generate() for _ in range(pop_size)]

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
        """Loop over evolutionary steps until get to a limit.

        Below is the Hyperparameters one can change to get better results.
        Just a quick note, all the problems have their specific parameters,
        it maybe mean the default will not work for one's case.

        Parameters:
        :epochs: Number of evolutions algorithm will loop over
        :pop_size: Number of Genomes (Genetic Representation of a solution)
        :selection: Selection funtion to be used (See Steps Module)
        :crossover: Crossover funtion to be used (See Steps Module)
        :multation: Multation funtion to be used (See Steps Module)
        :rng: A random.Random or an int seed, for reproducible runs.

        Optional Parameters:
        Depending on the function one choses, it might come with optional
        parameters that can be set as parameters on this function. The code
        will automatically deal with it.
        """
        rng = make_rng(rng)
        # initialize history stats
        start = time()
        self.history[start] = {
            "runs": list(),
            "best": tuple(),
            "elapsed": 0,
            "args": {
                "epochs": epochs, "pop_size": pop_size,
                "selection": selection.__name__,
                "crossover": crossover.__name__,
                "mutation": mutation.__name__,
                "verbose": verbose, "kwargs": kwargs
            }
        }
        # initialize the population for this round
        population = self._pop_generator(pop_size)
        best_result = (population[0]), self.fitness_function(population[0])

        for i in range(epochs):
            # keep the k most fitted and repopulate with new ones
            parents = selection(
                population, self.fitness_function, rng=rng,
                direction=self.direction, **kwargs)
            # Cross Over the parents to get a better solution
            children = crossover(*parents[:2], rng=rng, **kwargs)
            # Populate the next generation
            population = [*parents, *children]
            population.extend(
                self._pop_generator(pop_size=pop_size-len(population)))
            # Mutate the population
            safe_mutation = reject(
                mutation, self._check_constraints, self.max_tries)
            mutated = []
            for genome in population:
                try:
                    mutated.append(safe_mutation(genome, rng=rng, **kwargs))
                except InfeasibleError:
                    # no feasible mutant: keep the parent if it is feasible
                    # (crossover children may not be), else draw a new one
                    mutated.append(
                        genome if self._check_constraints(genome)
                        else self._pop_generator(1)[0])
            population = mutated
            # sort the population according to their fitness
            population.sort(key=lambda x: oriented(
                self.fitness_function(x), self.direction))
            # print the partial results if verbose
            if verbose:
                print("Epoch %i got fitness %.2f" % (
                    i, self.fitness_function(population[0])), population[0])
            # save the partial run history
            self.history[start]['runs'].append((  # type: ignore
                population[0], self.fitness_function(population[0])))

            if better(self.fitness_function(population[0]), best_result[1],
                      self.direction):
                best_result = (
                    population[0], self.fitness_function(population[0]))

        # save final results before quit
        self.history[start]['best'] = best_result
        self.history[start]['elapsed'] = time() - start

        # when done the epochs, return the most fit and its fitness score
        return best_result
