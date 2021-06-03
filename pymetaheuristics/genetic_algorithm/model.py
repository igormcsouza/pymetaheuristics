from typing import List, Tuple
from time import time

from pymetaheuristics.genetic_algorithm.types import (
    ConstraintFunction, CrossOverFunction, FitnessFunction, Genome,
    GenomeGeneratorFunction, MutationFunction, SelectionFunction)
from pymetaheuristics.genetic_algorithm.steps.selections import (
    random_weighted_selection)
from pymetaheuristics.genetic_algorithm.steps.crossovers import (
    single_point_crossover)
from pymetaheuristics.genetic_algorithm.steps.multations import inter_mutation


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

    Remember the GA will minimize the fitness function, so if you model is for
    maximize, return the result * -1 (See Knapsack model on test folder). You
    may also look for a specific result, so what you are looking is to minimize
    the difference on the fitness function.
    """

    def __init__(
        self, fitness_function: FitnessFunction,
        genome_generator: GenomeGeneratorFunction,
        constraints: List[ConstraintFunction] = [lambda x: True]
    ):
        self.fitness_function = fitness_function
        self.genome_generator = genome_generator
        self.constraints = constraints
        self.history = {}

    def _pop_generator(self, pop_size: int) -> List[Genome]:
        """Generate a population of genomes."""
        # Initialize the population as an empty list
        population: List[Genome] = list()
        # Generate pop_size Genomes and append it to the population
        for _ in range(pop_size):
            accepted = False
            genome = None

            # Check if given genome is accepted on the contraints.
            while not accepted:
                genome = self.genome_generator()
                accepted = self._check_constraints(genome)

            if genome:
                population.append(genome)

        return population

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

        Optional Parameters:
        Depending on the function one choses, it might come with optional
        parameters that can be set as parameters on this function. The code
        will automatically deal with it.
        """
        # initialize history stats
        start = time()
        self.history[start] = {
            'runs': list(),
            'args': {
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
            parents = selection(population, self.fitness_function, **kwargs)
            # Cross Over the parents to get a better solution
            children = crossover(*parents[:2], **kwargs)
            # Populate the next generation
            population = [*parents, *children]
            population.extend(
                self._pop_generator(pop_size=pop_size-len(population)))
            # Mutate the population
            population = [mutation(genome, **kwargs) for genome in population]
            # Check if every genome is still accepted by contraints
            for idx, genome in enumerate(population):
                accepted = False
                while not accepted:
                    accepted = self._check_constraints(genome)
                    if not accepted:
                        genome = self.genome_generator()
                population[idx] = genome
            # sort the population according to their fitness
            population.sort(key=lambda x: self.fitness_function(x))
            # print the partial results if verbose
            if verbose:
                print("Epoch %i got fitness %.2f" % (
                    i, self.fitness_function(population[0])), population[0])
            # save the partial run history
            self.history[start]['runs'].append((
                population[0], self.fitness_function(population[0])))

            if self.fitness_function(population[0]) < best_result[1]:
                best_result = (
                    population[0], self.fitness_function(population[0]))

        # save final results before quit
        self.history[start]['best'] = best_result
        self.history[start]['elapsed'] = time() - start

        # when done the epochs, return the most fit and its fitness score
        return best_result
