import numpy as np
import matplotlib.pyplot
import csv
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import os



class Cart:
    def __init__(self):
        self.list_of_victims = self.read_vital_signals('../datasets/data_4000v/env_vital_signals.txt')

        self.qPA_data = [signals[0] for signals in self.list_of_victims.values()]
        self.pulse_data = [signals[1] for signals in self.list_of_victims.values()]
        self.respiratory_data = [signals[2] for signals in self.list_of_victims.values()]
        self.gravity_data = [signals[4] for signals in self.list_of_victims.values()]

        self.tree = None

    def read_vital_signals(self, file_path):
        vital_signals = {}
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, file_path), 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                victim_id = int(parts[0])
                signals = list(map(float, parts[3:8]))
                signals.append(int(parts[7]))
                vital_signals[victim_id] = signals
        return vital_signals

    def train_cart_system(self):
        data = []
        for i in range(len(self.qPA_data)):
            data.append([self.qPA_data[i], self.pulse_data[i], self.respiratory_data[i]])

        X = np.array(data)
        y = np.array(self.gravity_data)
        
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        clf = tree.DecisionTreeClassifier()
        clf.fit(X_train, y_train)

        self.tree = clf

        # Calculate predictions
        y_pred = clf.predict(X_test)

        # Calculate metrics
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f_measure = f1_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Save metrics and parameters to a text file
        self.save_metrics_and_params('metricas_arvore_classificador.txt', precision, recall, f_measure, accuracy, conf_matrix, clf.get_params())

    def compute_priority(self, qPA, pulso, frequencia_respiratoria):
        return self.tree.predict([[qPA, pulso, frequencia_respiratoria]])

    def save_metrics_and_params(self, filename, precision, recall, f_measure, accuracy, conf_matrix, params):
        with open(filename, 'w') as file:
            # Write metrics
            file.write(f'Precisão: {precision:.4f}\n')
            file.write(f'Recall: {recall:.4f}\n')
            file.write(f'F-measure: {f_measure:.4f}\n')
            file.write(f'Acurácia: {accuracy:.4f}\n')
            file.write('Matriz de Confusão:\n')
            np.savetxt(file, conf_matrix, fmt='%d')
            
            # Write model parameters
            file.write('\nParâmetros do Modelo:\n')
            for param, value in params.items():
                file.write(f'{param}: {value}\n')

    def save_model(self, filename):
        """Salva o modelo treinado em um arquivo"""
        if self.tree is not None:
            joblib.dump(self.tree, filename)
            print(f"Modelo salvo em {filename}")
        else:
            print("Nenhum modelo treinado para salvar.")

if __name__ == '__main__':
    logic = Cart()
    logic.train_cart_system()
    logic.save_model('arvore_modelo_classificador.pkl')