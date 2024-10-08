# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.

from collections import deque
import math
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map
from k_means import KMeans

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

class Explorer(AbstAgent):
    def __init__(self, env, config_file, resc, type, general_map, nome):
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)
        self.walk_stack = Stack()  # a stack to store the movements
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.time_to_comeback = math.ceil(self.TLIM * 0.6)  # set the time to come back to the base
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is the seq number of the victim,(x,y) the position, <vs> the list of vital signals
        self.NAME = nome
        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())

        # atributos para lógica de busca
        self.queue = deque()
        self.visited = set()
        self.new_direction = None
        self.trap = False
        self.type = type
        self.found_new_exploring_point = False
        self.new_exploring_point = (0, 0)

        # atributo pra unir os mapas
        self.general_map = general_map

        # atributos para lógica de voltar pra base
        self.is_coming_to_base = False
        self.nodes_distances_from_base = {}
        self.nodes_distances_from_base[(0,0)] = 0
        self.min_cost_to_get_back = 0

        #atributos para lógica de clustering
        self.Kmeans = KMeans()
        self.is_explorer = True

    def get_estimated_time_to_return(self):
        obstacles = self.check_walls_and_lim()
        min_time = float('inf')

        for direction, status in enumerate(obstacles):
            if status == VS.CLEAR:
                dx, dy = Explorer.AC_INCR[direction]
                neighbor_position = (self.x + dx, self.y + dy)

                #Procura menor tempo pra voltar entre os vizinhos
                if self.map.in_map(neighbor_position):
                    if self.nodes_distances_from_base[neighbor_position] < min_time:
                        min_time = self.nodes_distances_from_base[neighbor_position]
        
        self.min_cost_to_get_back = min_time

    def fill_nodes_distances_from_base(self, neighbour_position):
        """
            This function is called for each time the explorer is avaliating its neighbours.
            It fills the self.nodes_distances_from_base structure, that for each block visited (key),
            points to the distance to origin (value).
        """
        weight = self.map.get((self.x, self.y))[0]
        if neighbour_position not in self.nodes_distances_from_base:
            self.nodes_distances_from_base[neighbour_position] = self.nodes_distances_from_base[self.x, self.y] + weight
        elif self.nodes_distances_from_base[neighbour_position] > self.nodes_distances_from_base[self.x, self.y] + weight:
            self.nodes_distances_from_base[neighbour_position] = self.nodes_distances_from_base[(self.x, self.y)] + weight

    #checa se uma coordenada possui mais de 4 direções livres
    def has_more_than_four_directions(self):
        directions = self.check_walls_and_lim()
        unblocked_count = sum(1 for direction in directions if direction == VS.CLEAR)
        if unblocked_count >= 4:
            return True
        else:
            return False
    
    def get_clear_direction(self):
        directions = self.check_walls_and_lim()
        count = 0


        for direction, status in enumerate(directions):
            if status == VS.CLEAR:
                dx, dy = Explorer.AC_INCR[direction]
                new_position = (self.x + dx, self.y + dy)

                self.fill_nodes_distances_from_base(new_position)

                count += 1
                if count == self.type:
                    self.new_direction = direction

    #quando o robô ta andando em uma direção e bate em algo, ele deve explorar em volta de onde bateu
    #a "new_base" é a coordenada que ele vai explorar em volta
    def set_new_base(self):
        directions = self.check_walls_and_lim()
        if self.has_more_than_four_directions() == False and self.new_direction == None:
            return self.get_next_position()
        
        elif self.has_more_than_four_directions() == True:
            if self.new_direction == None:
                self.get_clear_direction()
                return Explorer.AC_INCR[self.new_direction]
            else:
                if directions[self.new_direction] == VS.CLEAR and self.found_new_exploring_point == False:
                    dx, dy = Explorer.AC_INCR[self.new_direction]
                    new_position = (self.x + dx, self.y + dy)
                    self.fill_nodes_distances_from_base(new_position)
                    return dx, dy
                else:
                    self.found_new_exploring_point = True
                    self.new_exploring_point = (self.x, self.y)
                    return self.get_next_position()   
        else:
            self.found_new_exploring_point = True
            self.new_exploring_point = (self.x, self.y)
            return self.get_next_position()

    #esse metodo faz o robo andar sempre nos pontos mais proximos da base designada a ele
    def check_direction(self):
        obstacles = self.check_walls_and_lim()
        min_distance = float('inf')
        best_direction = None

        for direction, status in enumerate(obstacles):
            if status == VS.CLEAR:
                dx, dy = Explorer.AC_INCR[direction]

                new_x = self.x + dx
                new_y = self.y + dy

                distance_to_origin = math.sqrt((new_x - self.new_exploring_point[0]) ** 2 + (new_y - self.new_exploring_point[1]) ** 2)

                if (new_x, new_y) not in self.visited and distance_to_origin < min_distance:
                    min_distance = distance_to_origin
                    best_direction = (dx, dy)
                
                self.fill_nodes_distances_from_base((new_x, new_y))


        if best_direction is not None:
            self.queue.append(best_direction)
            return best_direction
        else:
            return None
            
    def trapped(self):
        """
            Essa função é chamada quando o robo fica preso. Pra achar um caminho ele volta por onde andou (lista queue)
            até que encontre em alguma posição anterior um novo caminho livre
        """
        direction = self.check_direction()
        # Vai entrar aqui se achar um novo caminho
        if direction:
            self.trap = False
            return direction
        # Se não continua voltando
        if len(self.queue) == 0:
            dx, dy = (0,0)
            return dx, dy
        else:
            dx, dy = self.queue.pop()
            return -dx, -dy
    
    def get_next_position(self):
        # Checa se o robô está num canto em que tudo em volta é parede ou ja foi visitado
        if self.trap:
            return self.trapped()

        # O robô está preso se não entrar nesta condição
        if (self.x, self.y) not in self.visited:
            self.visited.add((self.x, self.y))
            direction = self.check_direction()
            if direction:
                return direction

        # Como ele está preso é necessário voltar
        self.trap = True
        dx, dy = self.queue.pop()
        return -dx, -dy

    def heuristic(node, base):
        return abs(node.x) + abs(node.y)

    def explore(self):   
        if self.found_new_exploring_point == False:
           dx, dy = self.set_new_base()
           
        else:
           dx, dy = self.get_next_position()

        # Moves the body to another position
        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()

        # Test the result of the walk action
        # Should never bump, but for safe functionning let's test
        if result == VS.BUMPED:
            # update the map with the wall
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
            print(f"{self.NAME}: Wall or grid limit reached at ({self.x + dx}, {self.y + dy})")

        if result == VS.EXECUTED:
            # check for victim returns -1 if there is no victim or the sequential
            # the sequential number of a found victim
            self.walk_stack.push((dx, dy))

            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy          

            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)
                print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                print(f"{self.NAME} Seq: {seq} Vital signals: {vs}")

            # Calculates the difficulty of the visited cell
            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
                difficulty = difficulty / self.COST_DIAG

            # Update the map with the new cell
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())

            print(f"{self.NAME}:at ({self.x}, {self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")

        return

    def come_back(self):

        if (self.x, self.y) == (0, 0):
            print(f"{self.NAME}: already in base!")
            return
        
        obstacles = self.check_walls_and_lim()

        min_distance = float('inf')
        best_direction = None

        #Basicamente sempre encontra o menor caminho pra voltar entre seus vizinhos
        for direction, status in enumerate(obstacles):
            if status == VS.CLEAR:
                dx, dy = Explorer.AC_INCR[direction]
                neighbor_position = (self.x + dx, self.y + dy)

                if self.map.in_map(neighbor_position):
                    if self.nodes_distances_from_base[neighbor_position] < min_distance:

                        #Se estiver na mesma linha ou na mesma coluna, anda sempre em linha
                        if dx == 0 or dy == 0:
                            min_distance = self.nodes_distances_from_base[neighbor_position] + (self.COST_LINE - self.COST_DIAG)
                        else:
                            min_distance = self.nodes_distances_from_base[neighbor_position] + (self.COST_DIAG - self.COST_LINE)
                        
                        best_direction = (dx, dy)

        dx, dy = best_direction

        result = self.walk(dx, dy)

        if result == VS.BUMPED:
            print(f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
            return
        
        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
        
    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""
        if self.get_rtime() > self.min_cost_to_get_back * 2:
            self.explore()
            self.get_estimated_time_to_return()
            return True
        else:
            # time to come back to the base
            if self.x == 0 and self.y == 0:
                # time to wake up the rescuer
                # pass the walls and the victims (here, they're empty)
                self.set_state(VS.IDLE)
                print(f"{self.NAME}: rtime {self.get_rtime()}, invoking the rescuer")
                self.resc.sync_explorers(self.map, self.victims)
                #input(f"{self.NAME}: type [ENTER] to proceed")

                #Do not save victims yet, wait for clusters
                #self.resc.go_save_victims(self.map, self.victims)
                return False
            else:
                self.come_back()
                return True

        