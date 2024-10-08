##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim
### Not a complete version of DFS; it comes back prematuraly
### to the base when it enters into a dead end position


import math
import os
import random
from map import Map
from vs.abstract_agent import AbstAgent
from vs.physical_agent import PhysAgent
from vs.constants import VS
import sys
from abc import ABC, abstractmethod
import heapq
import csv
from bfs import BFS
#from k_means import KMeans
from sklearn.cluster import KMeans

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstAgent):
    def __init__(self, env, config_file, nb_of_explorers=1,clusters=[]):
        """ 
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file"""

        super().__init__(env, config_file)
        self.config_file = config_file

        # Specific initialization for the rescuer
        self.nb_of_explorers = nb_of_explorers       # number of explorer agents to wait for start
        self.received_maps = 0                       # counts the number of explorers' maps
        self.map = Map()            # explorer will pass the map
        self.victims = {}         # list of found victims
        self.plan = []              # a list of planned actions
        self.plan_x = 0             # the x position of the rescuer during the planning phase
        self.plan_y = 0             # the y position of the rescuer during the planning phase
        self.plan_visited = set()   # positions already planned to be visited 
        self.plan_rtime = self.TLIM # the remaing time during the planning phase
        self.plan_walk_time = 0.0   # previewed time to walk during rescue
        self.x = 0                  # the current x position of the rescuer when executing the plan
        self.y = 0                  # the current y position of the rescuer when executing the plan
        self.clusters = clusters     # the clusters of victims this agent should take care of - see the method cluster_victims
        self.sequences = clusters    # the sequence of visit of victims for each cluster

        self.regressor = load_model("./modelo_rede_neural.h5")
        self.classifier = load_model("./modelo_rede_neural_classificador.h5")

                
        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)
        self.is_explorer = False

    def save_cluster_csv(self, cluster, cluster_id):
        filename = f"./clusters/cluster{cluster_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for vic_id, values in cluster.items():
                x, y = values[0]      # x,y coordinates
                vs = values[1]        # list of vital signals
                writer.writerow([vic_id, x, y, vs[6], vs[7]])

    def save_sequence_csv(self, sequence, sequence_id):
        filename = f"./clusters/seq{sequence_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for id, values in sequence.items():
                x, y = values[0]      # x,y coordinates
                vs = values[1]        # list of vital signals
                writer.writerow([id, x, y, vs[6], vs[7]])

    #TODO: arrumar cluster, chamar k_means
    def cluster_victims(self):
        """ this method does a naive clustering of victims per quadrant: victims in the
            upper left quadrant compose a cluster, victims in the upper right quadrant, another one, and so on.
            
            @returns: a list of clusters where each cluster is a dictionary in the format [vic_id]: ((x,y), [<vs>])
                      such as vic_id is the victim id, (x,y) is the victim's position, and [<vs>] the list of vital signals
                      including the severity value and the corresponding label"""
        N_CLUSTERS = 4
        coordinates = []
        for key, values in self.victims.items():  # values are pairs: ((x,y), [<vital signals list>])
            coordinates.append(values[0])
        kmeans = KMeans(n_clusters=N_CLUSTERS, max_iter=300)
        cluster_indexes = kmeans.fit_predict(coordinates)


        #Create clusters from positions
        clusters = [[] for _ in range(4)]
        final_clusters = []
        #0 -> coordenadas
        j = 0
        for i in cluster_indexes:
            clusters[i].append(coordinates[j])
            j += 1

        #mapeia coordenadas -> id
        victim_id_to_coordinates = {}
        for key, values in self.victims.items():
            victim_id_to_coordinates[values[0]] = key
        
        for cluster in clusters:
            cluter_tmp = {}
            for position in cluster:
                victim_id = victim_id_to_coordinates[position]
                cluter_tmp[victim_id] = self.victims[victim_id]
            final_clusters.append(cluter_tmp)

        return final_clusters

    def predict_severity_and_class(self):
        for vic_id, values in self.victims.items():
            _, vital_signals = values
            
            #Extract the last 3 vital signals
            last_three_signals = vital_signals[-3:]
            last_three_signals_array = np.array(last_three_signals).reshape(1, -1)
            
            #Predict severity value using the regressor
            severity_value = self.regressor.predict(last_three_signals_array)[0][0]
            
            #Predict severity class using the classifier
            severity_class_prob = self.classifier.predict(last_three_signals_array)
            severity_class = np.argmax(severity_class_prob) + 1
            
            #Append the predictions to the vital signals
            vital_signals.extend([severity_value, severity_class])
            self.victims[vic_id] = (values[0], vital_signals)
            print (self.victims[vic_id])

    def sequencing(self):
        """ Currently, this method sort the victims by the x coordinate followed by the y coordinate
            @TODO It must be replaced by a Genetic Algorithm that finds the possibly best visiting order """

        """ We consider an agent may have different sequences of rescue. The idea is the rescuer can execute
            sequence[0], sequence[1], ...
            A sequence is a dictionary with the following structure: [vic_id]: ((x,y), [<vs>]"""
        new_sequences = []
        for seq in self.sequences:
            new_sequences.append(seq)

        from sequencing import SenquencyDefiner
        
        #the list of sequences needs to have more than 1 entry to calculate the sequence to save the victims!
        if len(self.sequences[0]) > 1:
            #we consider only the first sequence
            sequency = SenquencyDefiner(self.sequences[0])
            self.sequences[0] = sequency.get_best_sequencing()
    
    def planner(self):
        """ A method that calculates the path between victims: walk actions in a OFF-LINE MANNER (the agent plans, stores the plan, and
            after it executes. Eeach element of the plan is a pair dx, dy that defines the increments for the the x-axis and  y-axis."""


        # let's instantiate the breadth-first search
        bfs = BFS(self.map, self.COST_LINE, self.COST_DIAG)

        # for each victim of the first sequence of rescue for this agent, we're going go calculate a path
        # starting at the base - always at (0,0) in relative coords
        
        if not self.sequences:   # no sequence assigned to the agent, nothing to do
            return

        # we consider only the first sequence (the simpler case)
        # The victims are sorted by x followed by y positions: [vic_id]: ((x,y), [<vs>]

        sequence = self.sequences[0]
        start = (0,0) # always from starting at the base
        for vic_id in sequence:
            goal = sequence[vic_id][0]
            plan, time = bfs.search(start, goal, self.plan_rtime)
            self.plan = self.plan + plan
            self.plan_rtime = self.plan_rtime - time
            start = goal

        # Plan to come back to the base
        goal = (0,0)
        plan, time = bfs.search(start, goal, self.plan_rtime)
        self.plan = self.plan + plan
        self.plan_rtime = self.plan_rtime - time

    def sync_explorers(self, explorer_map, victims):
        """ This method should be invoked only to the master agent

        Each explorer sends the map containing the obstacles and
        victims' location. The master rescuer updates its map with the
        received one. It does the same for the victims' vital signals.
        After, it should classify each severity of each victim (critical, ..., stable);
        Following, using some clustering method, it should group the victims and
        and pass one (or more)clusters to each rescuer """

        self.received_maps += 1

        print(f"{self.NAME} Map received from the explorer")
        self.map.set_map_data(explorer_map)
        self.victims.update(victims)

        if self.received_maps == self.nb_of_explorers:
            print(f"{self.NAME} all maps received from the explorers")
            #self.map.draw()
            #print(f"{self.NAME} found victims by all explorers:\n{self.victims}")

            #@TODO predict the severity and the class of victims' using a classifier
            self.predict_severity_and_class()

            #@TODO cluster the victims possibly using the severity and other criteria
            # Here, there 4 clusters
            clusters_of_vic = self.cluster_victims()

            for i, cluster in enumerate(clusters_of_vic):
                self.save_cluster_csv(cluster, i+1)    # file names start at 1
  
            # Instantiate the other rescuers
            rescuers = [None] * 4
            rescuers[0] = self                    # the master rescuer is the index 0 agent

            # Assign the cluster the master agent is in charge of 
            self.clusters = [clusters_of_vic[0]]  # the first one

            # Instantiate the other rescuers and assign the clusters to them
            for i in range(1, 4):
                #print(f"{self.NAME} instantianting rescuer {i+1}, {self.get_env()}")
                #filename = f"rescuer_{i+1:1d}_config.txt"
                #config_file = os.path.join(self.config_folder, filename)
                # each rescuer receives one cluster of victims
                rescuers[i] = Rescuer(self.get_env(), self.config_file, 4, [clusters_of_vic[i]]) 
                rescuers[i].map = self.map     # each rescuer have the map

            
            # Calculate the sequence of rescue for each agent
            # In this case, each agent has just one cluster and one sequence
            self.sequences = self.clusters         

            # For each rescuer, we calculate the rescue sequence 
            for i, rescuer in enumerate(rescuers):
                rescuer.sequencing()         # the sequencing will reorder the cluster
                
                for j, sequence in enumerate(rescuer.sequences):
                    if j == 0:
                        self.save_sequence_csv(sequence, i+1)              # primeira sequencia do 1o. cluster 1: seq1 
                    else:
                        self.save_sequence_csv(sequence, (i+1)+ j*10)      # demais sequencias do 1o. cluster: seq11, seq12, seq13, ...

            
                rescuer.planner()            # make the plan for the trajectory
                rescuer.set_state(VS.ACTIVE) # from now, the simulator calls the deliberation method 

    def deliberate(self) -> bool:
        """ This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do """

        # No more actions to do
        if self.plan == []:  # empty list, no more actions to do
           print(f"{self.NAME} has finished the plan [ENTER]")
           return False

        # Takes the first action of the plan (walk action) and removes it from the plan
        dx, dy = self.plan.pop(0)
        #print(f"{self.NAME} pop dx: {dx} dy: {dy} ")

        # Walk - just one step per deliberation
        walked = self.walk(dx, dy)

        # Rescue the victim at the current position
        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy
            #print(f"{self.NAME} Walk ok - Rescuer at position ({self.x}, {self.y})")

            # check if there is a victim at the current position
            if self.map.in_map((self.x, self.y)):
                vic_id = self.map.get_vic_id((self.x, self.y))
                if vic_id != VS.NO_VICTIM:
                    self.first_aid()
                    #if self.first_aid(): # True when rescued
                        #print(f"{self.NAME} Victim rescued at ({self.x}, {self.y})")                    
        else:
            print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.x})")
            
        return True








##########################################################
    # Esses métodos não são necessários (dá pra tirar depois que tiver tudo pronto)
    def calc_gravity(self, vs):
        sinais = vs[-3:]
        sinais_array = np.array(sinais).reshape(1, -1) 
        return self.model.predict(sinais_array)[0][0] 
    
    def a_star_search(self, start, goal):
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0
        
        while open_list:
            _, current = heapq.heappop(open_list)

            if current == goal:
                break

            neighbors = self.map.get_neighbors(current)

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                next_position = (current[0] + dx, current[1] + dy)

                if next_position in neighbors:
                    move_cost = 1 if dx == 0 or dy == 0 else 1.5
                    new_cost = cost_so_far[current] + move_cost

                    if next_position not in cost_so_far or new_cost < cost_so_far[next_position]:
                        cost_so_far[next_position] = new_cost
                        priority = new_cost + self.heuristic(next_position, goal)
                        heapq.heappush(open_list, (priority, next_position))
                        came_from[next_position] = current

        return came_from, cost_so_far

    def heuristic(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def reconstruct_path(self, came_from, start, goal):
        current = goal
        path = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path

    def go_save_victims(self, map, victims):
        self.map = map
        self.victims = victims
        gravity_list = []

        # Calcula a gravidade de cada vítima e armazena em uma lista
        for seq, data in self.victims.items():
            coord, vital_signals = data
            gravity = self.calc_gravity(vital_signals)
            gravity_list.append((gravity, seq, coord, vital_signals))

        # Ordena as vítimas pela gravidade (mais próxima de 0 primeiro)
        gravity_list.sort(key=lambda x: abs(x[0]))

        self.plan = []
        start_position = (0, 0)  # Começando da base

        for gravity, seq, coord, vital_signals in gravity_list:
            path, _ = self.a_star_search(start_position, coord)
            path_to_victim = self.reconstruct_path(path, start_position, coord)

            # Adiciona o caminho até a vítima no plano
            for position in path_to_victim[1:]:
                dx = position[0] - start_position[0]
                dy = position[1] - start_position[1]
                self.plan.append((dx, dy, False))
                start_position = position

            self.plan.append((0, 0, True))  # Resgata a vítima

        # Planeja o caminho de volta à base
        if start_position != (0, 0):
            path_to_base, _ = self.a_star_search(start_position, (0, 0))
            path_to_base = self.reconstruct_path(path_to_base, start_position, (0, 0))
            for position in path_to_base[1:]:
                dx = position[0] - start_position[0]
                dy = position[1] - start_position[1]
                self.plan.append((dx, dy, False))
                start_position = position

        self.plan.append((0, 0, True))  

        print(gravity_list)
        self.map.draw()

        print(f"{self.NAME} List of found victims received from the explorer")

        for seq, data in self.victims.items():
            coord, vital_signals = data
            x, y = coord
            self.calc_gravity(vital_signals)
            print(f"{self.NAME} Victim seq number: {seq} at ({x}, {y}) vs: {vital_signals}")

        print(f"{self.NAME} time limit to rescue {self.plan_rtime}")

        print(f"{self.NAME} PLAN")
        i = 1
        self.plan_x = 0
        self.plan_y = 0
        for a in self.plan:
            self.plan_x += a[0]
            self.plan_y += a[1]
            print(f"{self.NAME} {i}) dxy=({a[0]}, {a[1]}) vic: a[2] => at({self.plan_x}, {self.plan_y})")
            i += 1

        print(f"{self.NAME} END OF PLAN")
        print(self.plan)
        self.set_state(VS.ACTIVE)