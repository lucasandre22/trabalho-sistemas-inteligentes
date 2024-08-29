from tensorflow.keras.models import load_model
import os
import numpy as np

def read_vital_signals(file_path):
    sinais_virais = {}
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, file_path), 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            id_vitima = int(parts[0])
            sinais = list(map(float, parts[3:6])) 
            sinais_virais[id_vitima] = sinais
    return sinais_virais

def main():
    classifier = load_model("./modelo_rede_neural_classificador.h5")

    file_path = '../datasets/data_408v_94x94/env_vital_signals_cego.txt'
    
    sinais_vitais = read_vital_signals(file_path)
    qPA_data = [signals[0] for signals in sinais_vitais.values()]
    pulse_data = [signals[1] for signals in sinais_vitais.values()]
    respiratory_data = [signals[2] for signals in sinais_vitais.values()]

    X = np.array(list(zip(qPA_data, pulse_data, respiratory_data)))

    severity_class_prob = classifier.predict(X)

    print(f'Últimos três sinais vitais: {severity_class_prob}')
    print(f'Classe prevista: {severity_class_prob}')

if __name__ == "__main__":
    main()

#gerar arquivo pred.txt
#id, x, y, gravity, class