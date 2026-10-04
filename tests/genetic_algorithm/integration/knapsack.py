from random import randint

from pymetaheuristics.core import Direction, Problem, max_iterations
from pymetaheuristics.genetic_algorithm import genetic_algorithm

items = [
    [25, 1.2],
    [40, 7.6],
    [10, 2.5],
    [17, 1.5],
    [42, 1.1],
    [29, 3.1],
    [14, 0.5],
    [36, 3.5],
]


def genome_generator():
    genome = list()
    for _ in range(len(items)):
        genome.append(randint(0, 1))
    return genome


def fitness_function(genome):
    score = 0
    for i, digit in enumerate(genome):
        score += digit * items[i][1]
    return score


def maximun_capacity(genome):
    weight = 0
    for i, digit in enumerate(genome):
        weight += digit * items[i][0]
    return weight <= 100


problem = Problem(
    generate=genome_generator,
    evaluate=fitness_function,  # total value, maximized as is
    feasible=maximun_capacity,
    direction=Direction.MAXIMIZE
)

result = genetic_algorithm(
    problem, stop=max_iterations(30), rng=0, population_size=10, k=5)

print("Genetic Algorithm result", result.best_solution, result.best_value)
print("Ground Truth", ([0, 1, 1, 1, 0, 1, 0, 0], 14.7), sep="\n")

ans = (14.7 - round(result.best_value, 2)) / 14.7
print(round(ans*100, 2), "%... off the optimal")
