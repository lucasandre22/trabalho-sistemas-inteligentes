from tensorflow.keras.models import load_model
import os
import numpy as np

classifier = load_model("./modelo_rede_neural_classificador.h5")

def read_vital_signals(file_path):
    sinais_virais = {}
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, file_path), 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            id_vitima = int(parts[0])
            sinais = list(map(float, parts[3:8])) 
            sinais_virais[id_vitima] = sinais
    return sinais_virais

file_path = '../datasets/data_408v_94x94/env_vital_signals_cego.txt'
sinais_vitais = read_vital_signals(file_path)
last_three_signals = sinais_vitais[-3:]
last_three_signals_array = np.array(last_three_signals).reshape(1, -1)
#Predict severity value using the regressor
severity_value = classifier.predict(last_three_signals_array)[0][0]

#gerar arquivo pred.txt
#id, x, y, gravity, class