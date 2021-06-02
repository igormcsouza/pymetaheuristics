from pymetaheuristics.genetic_algorithm.steps import (
    mutation, random_weighted_selection, single_point_crossover)
from typing import List, Tuple
from pymetaheuristics.genetic_algorithm.types import (
    ConstraintFunction, CrossOverFunction, FitnessFunction, Genome,
    GenomeGeneratorFunction, MutationFunction, SelectionFunction)


class GeneticAlgorithm():

    def __init__(
        self, fitness_function: FitnessFunction,
        genome_generator: GenomeGeneratorFunction,
        constraints: List[ConstraintFunction] = None
    ):
        self.fitness_function = fitness_function
        self.genome_generator = genome_generator
        self.constraints = constraints

    def _pop_generator(self, pop_size) -> List[Genome]:
        population: List[Genome] = list()
        for _ in range(pop_size):
            genome = self.genome_generator()

            if self.constraints:
                accepted = False
                while not accepted:
                    for constraint in self.constraints:
                        accepted = constraint(genome)

                    genome = self.genome_generator()

            population.append(genome)

        return population

    def add_constraint(self, constraint: ConstraintFunction):
        """Genetic Contraint for a Gene."""
        if self.constraints:
            self.constraints.append(constraint)
        else:
            self.constraints = [constraint]

    def train(
        self,
        epochs: int,
        pop_size: int,
        selection: SelectionFunction = random_weighted_selection,
        crossover: CrossOverFunction = single_point_crossover,
        mutation: MutationFunction = mutation,
        verbose: bool = False,
        **kwargs
    ) -> Tuple[Genome, float]:
        """Fit the population through the epochs."""
        # initialize the population for this round
        population = self._pop_generator(pop_size)

        for i in range(epochs):
            # keep the 2 most fitted and repopulate with new ones
            parents = selection(population, self.fitness_function, **kwargs)
            # Cross Over the parents to get a better solution
            children = crossover(*parents)
            # Populate the next generation
            population = [*parents, *children] + self._pop_generator(
                pop_size=pop_size-len(population))
            # Mutate the population
            population = [mutation(genome, **kwargs) for genome in population]
            # sort the population according to their fitness
            population.sort(key=lambda x: self.fitness_function(x))
            # print the partial results if verbose
            if verbose:
                print("Epoch %i got fitness %.2f" % (
                    i, self.fitness_function(population[0])), population[0])
        # when done the epochs, return the most fit and its fitness score
        return population[0], self.fitness_function(population[0])
