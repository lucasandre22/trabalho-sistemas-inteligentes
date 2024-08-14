import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

class RedeNeural:
    def __init__(self, data_path):
        self.list_of_victims = self.read_vital_signals(data_path)

        self.qPA_data = [signals[0] for signals in self.list_of_victims.values()]
        self.pulse_data = [signals[1] for signals in self.list_of_victims.values()]
        self.respiratory_data = [signals[2] for signals in self.list_of_victims.values()]
        self.gravity_data = [signals[3] for signals in self.list_of_victims.values()]

        self.X = np.array(list(zip(self.qPA_data, self.pulse_data, self.respiratory_data)))
        self.y = np.array(self.gravity_data)

        self.model = Sequential()
        self.model.add(Dense(64, input_shape=(3,), activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(1))

        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    def read_vital_signals(self, file_path):
        vital_signals = {}
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, file_path), 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                victim_id = int(parts[0])
                signals = list(map(float, parts[3:7]))
                vital_signals[victim_id] = signals
        return vital_signals

    def train(self):
        self.model.fit(self.X, self.y, epochs=15, batch_size=1)

    def evaluate(self):
        loss, mse = self.model.evaluate(self.X, self.y)
        print(f'Loss: {loss}, MSE: {mse}')

    def predict(self, new_data):
        predicao_normalizada = self.model.predict(new_data)
        return predicao_normalizada

    def save(self):
        self.model.save("RNmodelo.h5")

if __name__ == "__main__":
    model = RedeNeural('../datasets/data_4000v/env_vital_signals.txt')

    model.train()

    model.evaluate()

    novos_dados = np.array([[-5.724860,46.693170,12.655292]])
    classe_predita = model.predict(novos_dados)
    print(f'Classe predita: {classe_predita}')
