from random import shuffle

from pymetaheuristics.utils.distances import euclidian_distance
from pymetaheuristics.genetic_algorithm.steps import pmx_single_point
from pymetaheuristics.genetic_algorithm.model import GeneticAlgorithm
from pymetaheuristics.genetic_algorithm.types import Genome


cities_list = [
    [42.5, 48.9],
    [97.2, 32.1],
    [23.5, 85.9],
    [32.8, 45.2],
    [12.5, 69.9]
]

distance_matrix = list()
for i, city1 in enumerate(cities_list):
    distance_matrix.append([])
    for city2 in cities_list:
        distance_matrix[i].append(euclidian_distance(city1, city2))


def genome_generator() -> Genome:
    sequence = list(range(len(cities_list)))
    shuffle(sequence)
    return sequence


def fitness_function(genome: Genome) -> float:
    fitness = 0
    for i in range(len(genome)):
        fitness += distance_matrix[genome[i]][genome[i-1]]

    return fitness


model = GeneticAlgorithm(
    fitness_function=fitness_function,
    genome_generator=genome_generator,
    constraints=[]
)

result = model.train(epochs=15, pop_size=10, crossover=pmx_single_point)

print("Genetic Algorithm result", result, sep="\n")
print("Ground Truth", ([2, 4, 3, 0, 1], 210.24), sep="\n")

ans = (round(result[1], 2) - 210.24) / 210.24
print(round(ans*100, 2), "%... off the optimal")
