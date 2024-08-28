import os
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

class ArvoreIndutivaRegressao:
    def __init__(self):
        self.list_of_victims = self.read_vital_signals('../datasets/data_4000v/env_vital_signals.txt')

        self.qPA_data = [signals[0] for signals in self.list_of_victims.values()]
        self.pulse_data = [signals[1] for signals in self.list_of_victims.values()]
        self.respiratory_data = [signals[2] for signals in self.list_of_victims.values()]
        self.gravity_data = [signals[3] for signals in self.list_of_victims.values()]
        self.tree = None

        # Divide os dados em treino e teste
        self.X = np.array(list(zip(self.qPA_data, self.pulse_data, self.respiratory_data)))
        self.y = np.array(self.gravity_data)
        self.X_treino, self.X_teste, self.y_treino, self.y_teste = train_test_split(
            self.X, self.y, test_size=0.25, random_state=10)
        
    def read_vital_signals(self, file_path):
        vital_signals = {}
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, file_path), 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                victim_id = int(parts[0])
                signals = list(map(float, parts[3:7]))
                signals.append(int(parts[7]))
                vital_signals[victim_id] = signals
        return vital_signals
    
    def train_cart_system(self):
        regressor = DecisionTreeRegressor()
        regressor.fit(self.X_treino, self.y_treino)
        self.tree = regressor

    def compute_priority(self, qPA, pulso, frequencia_respiratoria):
        prediction = self.tree.predict([[qPA, pulso, frequencia_respiratoria]])[0]
        return max(0, min(100, prediction))

    def avaliar(self):
        previsoes_teste = self.tree.predict(self.X_teste)
        mse_teste = mean_squared_error(self.y_teste, previsoes_teste)
        rmse_teste = np.sqrt(mse_teste)
        
        print(f'Teste MSE: {mse_teste}, Teste RMSE: {rmse_teste}')
        
        return rmse_teste

    def salvar_metricas(self, rmse_teste):
        metricas = (
            f"RMSE Teste: {rmse_teste}\n"
        )

        with open('metricas_arvore_regressor.txt', 'w') as file:
            file.write("\nMétricas:\n")
            file.write(metricas)
        
    def save_model(self, caminho_modelo):
        with open(caminho_modelo, 'wb') as file:
            pickle.dump(self.tree, file)

if __name__ == '__main__':
    logic = ArvoreIndutivaRegressao()

    logic.train_cart_system()

    rmse_teste = logic.avaliar()

    logic.salvar_metricas(rmse_teste)

    logic.save_model('arvore_modelo_regressao.pkl')
