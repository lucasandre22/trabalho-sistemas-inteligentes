import pygad
import numpy
import csv
import math

#o conjunto dos agentes socorristas (𝐴𝑠) deve definir a sequência de salvamento das 
#vítimas para cada cluster por meio de um Algoritmo Genético (AG). 

class SenquencyDefiner():
    def __init__(self, initial_victims_sequence):
        
        #A sequence is a dictionary with the following structure: [vic_id]: ((x,y), [<vs>]
        #transform the structure to list, in order to adjust indexes and not map by vic_id
        self.initial_sequence = list(initial_victims_sequence.values())
        self.initial_sequence_map = initial_victims_sequence

    def save_sequence_csv(self, sequence, sequence_id):
        filename = f"./clusters/seq{sequence_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for id, values in sequence.items():
                x, y = values[0]      # x,y coordinates
                vs = values[1]        # list of vital signals
                writer.writerow([id, x, y, vs[6], vs[7]])
                
    def distance(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    
    #Returns how accurate the candidate is (how close from the ideal solution)
    def fitness_func(self, ga_instance, solution, solution_idx):
        total_distance = 0
        #total number of victims in the seq
        num_victims = len(solution)
        
        #Origin as the initial point
        initial_point = [(0, 0)]
        previous_victim = initial_point
        
        #for each victim, calculates the fitness value (percentage) for it
        for i in range(num_victims):
            current_victim = self.initial_sequence[int(solution[i])]
            #Do we use the severity or class_severity??
            severity = current_victim[1][6]
            class_severity = current_victim[1][7]
            total_distance += self.distance(previous_victim[0], current_victim[0]) / class_severity #exponential to make the severity change more the result
            previous_victim = current_victim
            
        return 1.0 / total_distance
    
    def get_best_sequencing(self):
        #number out of my butt
        num_generations = len(self.initial_sequence) * 5
        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=10,
            fitness_func=self.fitness_func,
            sol_per_pop=100,
            num_genes=len(self.initial_sequence),
            gene_type=int,
            init_range_low=0,
            init_range_high=len(self.initial_sequence) - 1,
            parent_selection_type="tournament",
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=10,
            gene_space=range(len(self.initial_sequence)),
            save_best_solutions=False,
            #parametro que salvou a vida -> https://pygad.readthedocs.io/en/latest/pygad_more.html#limit-the-gene-value-range-using-the-gene-space-parameter
            allow_duplicate_genes=False
        )

        ga_instance.run()
        best_solution, best_solution_fitness, solution_idx = ga_instance.best_solution()

        print("Best solution:", best_solution)
        print("Best fitness:", best_solution_fitness)
        
        #Ordena a lista de vítimas com base na melhor solução encontrada
        sorted_victims = [self.initial_sequence[int(idx)] for idx in best_solution]
        
        to_be_returned = {}
        
        #ineficiente quadrática, mas é o que temos visto as circunstâncias :)
        for j in sorted_victims:
            for vic_id, values in self.initial_sequence_map.items():
                if values == j:
                    to_be_returned[vic_id] = values

        return to_be_returned