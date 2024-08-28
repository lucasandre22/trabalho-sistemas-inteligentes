import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

class RedeNeural:
    def __init__(self, caminho_dados, 
                 numero_neuronios=64, 
                 numero_camadas=8, 
                 ativacao='relu', 
                 otimizador='adam', 
                 perda='mean_squared_error', 
                 metricas=['mean_squared_error'], 
                 epocas=80, 
                 tamanho_lote=3):
        
        self.list_of_victims = self.read_vital_signals(caminho_dados)

        self.qPA_data = [signals[0] for signals in self.list_of_victims.values()]
        self.pulse_data = [signals[1] for signals in self.list_of_victims.values()]
        self.respiratory_data = [signals[2] for signals in self.list_of_victims.values()]
        self.gravity_data = [signals[3] for signals in self.list_of_victims.values()]

        self.X = np.array(list(zip(self.qPA_data, self.pulse_data, self.respiratory_data)))
        self.y = np.array(self.gravity_data)

        # parâmetros
        self.numero_neuronios = numero_neuronios
        self.numero_camadas = numero_camadas
        self.ativacao = ativacao
        self.otimizador = otimizador
        self.perda = perda
        self.metricas = metricas
        self.epocas = epocas
        self.tamanho_lote = tamanho_lote

        # divide treino (75%) e teste (25%)
        self.X_treino, self.X_teste, self.y_treino, self.y_teste = train_test_split(
            self.X, self.y, test_size=0.25, random_state=10)  # random_state é só uma seed pra pegar os dados aleatórios
        
    def read_vital_signals(self, file_path):
        sinais_virais = {}
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, file_path), 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                id_vitima = int(parts[0])
                sinais = list(map(float, parts[3:7]))
                sinais_virais[id_vitima] = sinais
        return sinais_virais

    def build_model(self):
        model = Sequential()
        model.add(Dense(self.numero_neuronios, input_shape=(3,), activation=self.ativacao))
        
        for _ in range(self.numero_camadas - 1):
            model.add(Dense(self.numero_neuronios, activation=self.ativacao))
        
        model.add(Dense(1))
        
        model.compile(optimizer=self.otimizador, loss=self.perda, metrics=self.metricas)
        return model

    def train(self):
        model = self.build_model()
        model.fit(self.X_treino, self.y_treino, epochs=self.epocas, batch_size=self.tamanho_lote, verbose=0)
        return model

    def evaluate(self, model):
        previsoes_treino = model.predict(self.X_treino).flatten()
        mse_treino = mean_squared_error(self.y_treino, previsoes_treino)
        mae_treino = mean_absolute_error(self.y_treino, previsoes_treino)
        rmse_treino = np.sqrt(mse_treino)

        previsoes_teste = model.predict(self.X_teste).flatten()
        mse_teste = mean_squared_error(self.y_teste, previsoes_teste)
        mae_teste = mean_absolute_error(self.y_teste, previsoes_teste)
        rmse_teste = np.sqrt(mse_teste)

        print(f'Treino MSE: {mse_treino}, Treino MAE: {mae_treino}, Treino RMSE: {rmse_treino}')
        print(f'Teste MSE: {mse_teste}, Teste MAE: {mae_teste}, Teste RMSE: {rmse_teste}')

        if rmse_treino > rmse_teste:
            print("Possível underfitting")
        elif rmse_treino < rmse_teste:
            print("Possível overfitting")
        else:
            print("Modelo bem ajustado")

        return rmse_treino, rmse_teste

    def predict(self, model, novos_dados):
        predicao_normalizada = model.predict(novos_dados)
        return predicao_normalizada
    
    def salvar_modelo(self, model, caminho_modelo):
        model.save(caminho_modelo)
        print(f"Modelo salvo em: {caminho_modelo}")

    def salvar_metricas(self, rmse_treino, rmse_teste):
        parametros = (
            f"Numero de Neurônios: {self.numero_neuronios}\n"
            f"Numero de Camadas: {self.numero_camadas}\n"
            f"Ativação: {self.ativacao}\n"
            f"Otimizador: {self.otimizador}\n"
            f"Perda: {self.perda}\n"
            f"Metricas: {', '.join(self.metricas)}\n"
            f"Épocas: {self.epocas}\n"
            f"Tamanho do Lote: {self.tamanho_lote}\n"
        )
        
        metricas = (
            f"RMSE Treino: {rmse_treino}\n"
            f"RMSE Teste: {rmse_teste}\n"
        )

        with open('metricas_redeneural_regressor.txt', 'w') as file:
            file.write("Parâmetros do Modelo:\n")
            file.write(parametros)
            file.write("\nMétricas:\n")
            file.write(metricas)
        
if __name__ == "__main__":
    instancia_modelo = RedeNeural('../datasets/data_4000v/env_vital_signals.txt')

    modelo_treinado = instancia_modelo.train()

    rmse_treino, rmse_teste = instancia_modelo.evaluate(modelo_treinado)

    instancia_modelo.salvar_metricas(rmse_treino, rmse_teste)

    novos_dados = np.array([[-0.000000, 108.934128, 14.587328]]) 
    predicao = instancia_modelo.predict(modelo_treinado, novos_dados)

    instancia_modelo.salvar_modelo(modelo_treinado, 'modelo_rede_neural.h5')
    print(f'Predição: {predicao}')
