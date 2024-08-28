import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.optimizers import Adam


class RedeNeuralClassificador:
    def __init__(self, data_path, 
                 numero_neuronios=64, 
                 numero_camadas=8, 
                 ativacao='relu', 
                 otimizador='adam', 
                 taxa_aprendizado=0.002,  #Adicionado o parâmetro taxa_aprendizado
                 perda='categorical_crossentropy', 
                 metricas=['accuracy'], 
                 epocas=80, 
                 tamanho_lote=3):
        
        self.numero_neuronios = numero_neuronios
        self.numero_camadas = numero_camadas
        self.ativacao = ativacao
        self.otimizador = otimizador
        self.taxa_aprendizado = taxa_aprendizado  #Armazena a taxa de aprendizado
        self.perda = perda
        self.metricas = metricas
        self.epocas = epocas
        self.tamanho_lote = tamanho_lote
        #Leitura dos sinais vitais
        self.list_of_victims = self.read_vital_signals(data_path)

        #Extração dos dados de entrada e saída
        self.qPA_data = [signals[0] for signals in self.list_of_victims.values()]
        self.pulse_data = [signals[1] for signals in self.list_of_victims.values()]
        self.respiratory_data = [signals[2] for signals in self.list_of_victims.values()]
        self.gravity_data = [signals[4] for signals in self.list_of_victims.values()]  

        #Preparação dos dados de entrada (X) e saída (y)
        self.X = np.array(list(zip(self.qPA_data, self.pulse_data, self.respiratory_data)))
        self.y = np.array(self.gravity_data)

        #Divisão dos dados em 75% treino e 25% validação
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=0.25, random_state=42
        )
        
    def build_model(self):
        #Configura o otimizador com a taxa de aprendizado
        if self.otimizador == 'adam':
            optimizer = Adam(learning_rate=self.taxa_aprendizado)
        else:
            optimizer = self.otimizador
        
        #Construção do modelo
        model = Sequential()
        model.add(Input(shape=(3,)))

        for _ in range(self.numero_camadas):
            model.add(Dense(self.numero_neuronios, activation=self.ativacao))

        model.add(Dense(4, activation='softmax'))
        model.compile(optimizer=optimizer, loss=self.perda, metrics=self.metricas)
        return model
        
    #Método para ler o arquivo de sinais vitais
    def read_vital_signals(self, file_path):
        sinais_virais = {}
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, file_path), 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                id_vitima = int(parts[0])
                sinais = list(map(float, parts[3:8]))  #Ajuste na extração dos sinais
                sinais_virais[id_vitima] = sinais
        return sinais_virais
    
    def train(self):
        self.model = self.build_model()
        self.model.fit(self.X_train, to_categorical(self.y_train - 1, num_classes=4),
                       validation_data=(self.X_val, to_categorical(self.y_val - 1, num_classes=4)),
                       epochs=self.epocas, batch_size=self.tamanho_lote)

    #Método para avaliar o modelo
    def evaluate(self):
        #Previsões e métricas para o conjunto de validação
        previsoes_val = self.model.predict(self.X_val)
        previsoes_val_classes = np.argmax(previsoes_val, axis=1) + 1
        
        precision = precision_score(self.y_val, previsoes_val_classes, average='weighted')
        recall = recall_score(self.y_val, previsoes_val_classes, average='weighted')
        f_measure = f1_score(self.y_val, previsoes_val_classes, average='weighted')
        accuracy = accuracy_score(self.y_val, previsoes_val_classes)
        conf_matrix = confusion_matrix(self.y_val, previsoes_val_classes)

        #Salvar as métricas e parâmetros em um arquivo de texto
        self.save_metrics('metricas_redeneural_classificador.txt', precision, recall, f_measure, accuracy, conf_matrix)
    
    #Método para salvar métricas e parâmetros em um arquivo de texto
    def save_metrics(self, filename, precision, recall, f_measure, accuracy, conf_matrix):
        with open(filename, 'w') as file:
            file.write(f'Precisão: {precision:.4f}\n')
            file.write(f'Recall: {recall:.4f}\n')
            file.write(f'F-measure: {f_measure:.4f}\n')
            file.write(f'Acurácia: {accuracy:.4f}\n')
            file.write('Matriz de Confusão:\n')
            np.savetxt(file, conf_matrix, fmt='%d')
            file.write('\nParâmetros do Modelo:\n')
            file.write(f'Número de Neurônios: {self.numero_neuronios}\n')
            file.write(f'Número de Camadas: {self.numero_camadas}\n')
            file.write(f'Ativação: {self.ativacao}\n')
            file.write(f'Otimização: {self.otimizador}\n')
            file.write(f'Perda: {self.perda}\n')
            file.write(f'Métricas: {", ".join(self.metricas)}\n')
            file.write(f'Épocas: {self.epocas}\n')
            file.write(f'Tamanho do Lote: {self.tamanho_lote}\n')

    #Método para fazer predições com novos dados
    def predict(self, new_data):
        prediction = self.model.predict(new_data)
        predicted_class = np.argmax(prediction) + 1  #Ajuste para classes 1, 2, 3, 4
        return predicted_class

    #Método para salvar o modelo treinado
    def salvar_modelo(self, model_path):
        self.model.save(model_path)

if __name__ == "__main__":
    instancia_modelo = RedeNeuralClassificador('../datasets/data_4000v/env_vital_signals.txt')

    #Treinamento e avaliação do modelo
    instancia_modelo.train()
    instancia_modelo.evaluate()

    #Predição com novos dados
    novos_dados = np.array([[0.0, 108.934128, 14.587328]]) 
    classe_predita = instancia_modelo.predict(novos_dados)

    instancia_modelo.salvar_modelo('modelo_rede_neural_classificador.h5')
    print(f'Classe predita: {classe_predita}')