from typing import List, Tuple
from pymetaheuristics.genetic_algorithm.types import (
    ConstraintFunction, FitnessFunction, Genome, GenomeGeneratorFunction)


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

    def train(self, epochs: int, pop_size: int) -> Tuple[Genome, float]:
        # initialize the population for this round
        population = self._pop_generator(pop_size)

        for i in range(epochs):
            # sort the population according to their fitness
            population.sort(key=lambda x: self.fitness_function(x))
            # print the partial results
            print("Epoch %i got fitness %.2f" % (
                i, self.fitness_function(population[0])), population[0])
            # keep the 2 most fitted and repopulate with new ones
            population = population[:2] + self._pop_generator(
                pop_size=pop_size-2)

        # when done the epochs, return the most fit and its fitness score
        return population[0], self.fitness_function(population[0])
