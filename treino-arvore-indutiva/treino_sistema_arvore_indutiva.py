import numpy as np
import matplotlib.pyplot
import csv
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import os

class Cart:
    def __init__(self):
        
        self.list_of_victims = self.read_vital_signals('../datasets/data_4000v/env_vital_signals.txt')

        self.qPA_data = [signals[2] for signals in self.list_of_victims.values()]
        self.pulse_data = [signals[3] for signals in self.list_of_victims.values()]
        self.respiratory_data = [signals[4] for signals in self.list_of_victims.values()]
        self.gravity_data = [signals[5] for signals in self.list_of_victims.values()]

        self.tree = None

    # Le o TXT
    def read_vital_signals(self, file_path):
        vital_signals = {}
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, file_path), 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                victim_id = int(parts[0])
                signals = list(map(float, parts[1:6]))
                signals.append(int(parts[7]))
                vital_signals[victim_id] = signals
        return vital_signals
    
    def train_cart_system(self):
        data = []
        for i in range(0, len(self.qPA_data)):
            data.append([self.qPA_data[i], self.pulse_data[i], self.respiratory_data[i]])

        clf = tree.DecisionTreeClassifier()
        clf.fit(data, self.gravity_data)

        self.tree = clf

    def compute_priority(self, qPA, pulso, frequencia_respiratoria):
        return self.tree.predict([[qPA, pulso, frequencia_respiratoria]])

    def save_preditions(self, filename):
        """
            Salva arquivo CSV de treinamento
        """
        for cluster in self.get_clusters():
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['id','x','y','grav','classe'])
                for i in range(0, len(cluster)):
                    id = 1
                    x = 0
                    y = 0
                    grav = 1
                    classe = 1
                    writer.writerow([id, x, y, grav, classe])

if __name__ == '__main__':
    logic = Cart()

    # Treina
    logic.train_cart_system()

    assert logic.compute_priority(8.733333,135.824333,12.787053) == 2, "8.733333,135.824333,12.787053 should be 2"
    assert logic.compute_priority(8.733333,135.824333,12.787053) == 2, "8.733333,135.824333,12.787053 should be 2"
    assert logic.compute_priority(-0.000000,57.527259,14.500449) == 4, "-0.000000,57.527259,14.500449 should be 4"

    print(logic.compute_priority(3,14.527259,1.500449)) #