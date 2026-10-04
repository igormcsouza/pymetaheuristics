from random import shuffle

from pymetaheuristics.core import Problem, max_iterations
from pymetaheuristics.utils.distances import euclidian_distance
from pymetaheuristics.genetic_algorithm import genetic_algorithm
from pymetaheuristics.genetic_algorithm.steps.crossovers import (
    pmx_single_point)
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


problem = Problem(generate=genome_generator, evaluate=fitness_function)

result = genetic_algorithm(
    problem, stop=max_iterations(5), rng=0, population_size=10,
    crossover=pmx_single_point)

print("Genetic Algorithm result", result.best_solution, result.best_value)
print("Ground Truth", ([2, 4, 3, 0, 1], 210.24), sep="\n")

ans = (round(result.best_value, 2) - 210.24) / 210.24
print(round(ans*100, 2), "%... off the optimal")
