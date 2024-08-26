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
from abc import ABC, abstractmethod
import heapq

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstAgent):
    def __init__(self, env, config_file):
        """ 
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.map = None             # explorer will pass the map
        self.victims = None         # list of found victims
        self.plan = []              # a list of planned actions
        self.plan_x = 0             # the x position of the rescuer during the planning phase
        self.plan_y = 0             # the y position of the rescuer during the planning phase
        self.plan_visited = set()   # positions already planned to be visited 
        self.plan_rtime = self.TLIM # the remaing time during the planning phase
        self.plan_walk_time = 0.0   # previewed time to walk during rescue
        self.x = 0                  # the current x position of the rescuer when executing the plan
        self.y = 0                  # the current y position of the rescuer when executing the plan

        self.model = load_model("./modelo_rede_neural.h5")

                
        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)
        self.is_explorer = False


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


    def deliberate(self) -> bool:
        """ This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do """

        # Não há mais ações para fazer
        if not self.plan:  # lista vazia, nenhuma ação a ser feita
            input(f"{self.NAME} has finished the plan [ENTER]")
            return False

        # Toma a primeira ação do plano (ação de caminhar) e remove-a do plano
        dx, dy, there_is_vict = self.plan.pop(0)
        print(f"{self.NAME} pop dx: {dx} dy: {dy} vict: {there_is_vict}")
        print(self.plan)
        # Caminhar - apenas um passo por deliberação
        walked = self.walk(dx, dy)

        # Resgata a vítima na posição atual
        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy
            print(f"{self.NAME} Walk ok - Rescuer at position ({self.x}, {self.y})")
            # verifica se há uma vítima na posição atual
            if there_is_vict:
                rescued = self.first_aid()  # True quando resgatado
                if rescued:
                    print(f"{self.NAME} Victim rescued at ({self.x}, {self.y})")
                else:
                    print(f"{self.NAME} Plan fail - victim not found at ({self.x}, {self.y})")
        else:
            print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.y})")
        
        # Espera o usuário apertar ENTER antes de continuar para a próxima ação
        input(f"{self.NAME} remaining time: {self.get_rtime()} [Press ENTER to continue]")
        
        return True

