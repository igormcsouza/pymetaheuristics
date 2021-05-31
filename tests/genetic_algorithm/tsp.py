from random import randint, shuffle
from pymetaheuristics.utils.distances import euclidian_distance
from pymetaheuristics.genetic_algorithm.model import GeneticAlgorithm
from pymetaheuristics.genetic_algorithm.types import Genome


cities_list = list()
for _ in range(5):
    cities_list.append([randint(0, 1000), randint(0, 1000)])

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

result = model.train(5, 10)

print(result)
