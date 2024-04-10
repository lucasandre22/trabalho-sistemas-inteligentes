from collections import deque
import math
import heapq
from vs.abstract_agent import AbstAgent
from vs.constants import VS
import random
import csv

class KMeans:
    def __init__(self, n_clusters=4, max_iters=10):
        self.n_clusters = n_clusters 
        self.max_iters = max_iters
        self.clusters = None

    def get_clusters(self):
        return self.clusters

    def fit(self, coordinates):
        
        #Inicializa os centróides em alguma vítima aleatória
        self.centroids = [random.choice(coordinates) for _ in range(self.n_clusters)]

        for _ in range(self.max_iters):
            #Atribui cada ponto ao centróide mais próximo
            clusters = self.set_point_to_clusters(coordinates)

            #Atualiza os centróides com a média dos pontos em cada cluster
            new_centroids = [self.compute_centroid(cluster) for cluster in clusters]

            #Verifica se os centróides permaneceram os mesmos
            if self.centroids_converged(self.centroids, new_centroids):
                break  # Se convergiu, interrompe o loop

            self.centroids = new_centroids  #Atualiza os centróides para a próxima iteração
        self.clusters = clusters

    def set_point_to_clusters(self, coordinates):
        #cria uma lista de lista para os clusters
        clusters = [[] for _ in range(self.n_clusters)]

        # procura qual o centroide mais próximo pra cada coordenada da vítima
        for point in coordinates:  
            min_distance = float('inf')
            closest_centroid = None
            for i, centroid in enumerate(self.centroids):
                distance = self.euclidean_distance(point, centroid)
                if distance < min_distance:
                    min_distance = distance
                    closest_centroid = i
            clusters[closest_centroid].append(point)
            
        #retorna uma lista com os clusters
        return clusters

    def euclidean_distance(self, p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def compute_centroid(self, cluster):
        # Calcula o centróide de um cluster como a média dos pontos no cluster
        num_points = len(cluster)  # Número de pontos no cluster
        centroid = [sum(coords) / num_points for coords in zip(*cluster)]  # Calcula a média de cada coordenada
        return centroid

    def centroids_converged(self, centroids_old, centroids_new):
        # Verifica se os centróides antigos e novos são os mesmos
        return all(old == new for old, new in zip(centroids_old, centroids_new))


    def generate_clusters_files(self, victims_key, gravity, label):
        """
            Gera N arquivos clusterN.txt sendo N o total de clusters, para cada um,
            contendo as informações das vítimas.
        """
        
        number = 1
        for cluster in self.get_clusters():
            filename = "cluster" + str(number) + ".txt"
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['id','x','y','0.0','1'])
                for i in range(0, len(cluster)):
                    id = victims_key[cluster[i]]
                    x = cluster[i][0]
                    y = cluster[i][1]
                    gravity_ = gravity[id]
                    label_ = label[id]
                    writer.writerow([id, x, y, gravity_, label_])
            number += 1
