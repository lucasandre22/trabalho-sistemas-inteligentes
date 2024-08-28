import pygad
import numpy as np
import math

#Representação das vítimas
victims = [
    (203, 0, -4, 60.002298, 3),
    (202, 0, -7, 18.14836, 1),
    (201, 0, -11, 13.243103, 1),
    (207, 1, -17, 40.0, 2),
    (197, -1, -17, 78.851558, 4),
    (191, -3, -16, 13.404359, 1),
    (213, 3, -13, 63.965998, 3),
]

def distance(v1, v2):
    return math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)

def fitness_func(ga_instance, solution, solution_idx):
    total_distance = 0
    initial_point = (0, 0)
    previous_point = initial_point

    for i in range(len(solution)):
        current_victim = victims[int(solution[i])]
        current_point = (current_victim[1], current_victim[2])
        total_distance += distance(previous_point, current_point) / current_victim[4]
        previous_point = current_point

    return 1.0 / total_distance

def on_generation(ga):
    print("Generation", ga.generations_completed)
    print(ga.population)

ga_instance = pygad.GA(
    num_generations=500,
    num_parents_mating=20,
    fitness_func=fitness_func,
    sol_per_pop=100,
    num_genes=len(victims),
    gene_type=int,
    init_range_low=0,
    init_range_high=len(victims) - 1,
    gene_space=range(len(victims)),
    parent_selection_type="tournament",
    crossover_type="single_point",
    mutation_type="random",
    on_generation=on_generation,
    mutation_percent_genes=10,
    allow_duplicate_genes=False
)

ga_instance.run()
best_solution, best_solution_fitness, _ = ga_instance.best_solution()

sorted_victims = [victims[int(idx)] for idx in best_solution]

print("Best solution (indices):", best_solution)
print("Best fitness:", best_solution_fitness)
print("Sorted victims by best solution:")
for victim in sorted_victims:
    print(victim)
