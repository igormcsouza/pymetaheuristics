from random import randint

from pymetaheuristics.genetic_algorithm.model import GeneticAlgorithm

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
    return score * -1


def maximun_capacity(genome):
    weight = 0
    for i, digit in enumerate(genome):
        weight += digit * items[i][0]
    return weight <= 100


model = GeneticAlgorithm(
    genome_generator=genome_generator,
    fitness_function=fitness_function
)

model.add_constraint(maximun_capacity)

result = model.train(30, 10, k=5, verbose=True)

print("Genetic Algorithm result", result, sep="\n")
print("Ground Truth", ([0, 1, 1, 1, 0, 1, 0, 0], -14.7), sep="\n")

ans = (14.7 - abs(round(result[1], 2))) / 14.7
print(round(ans*100, 2), "%... off the optimal")
