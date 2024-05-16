import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot
class FuzzyLogic:
    def __init__(self):
        
        self.list_of_victims = self.read_vital_signals('./env_vital_signals.txt')
        self.qPA_var = None
        self.pulso_var = None
        self.frequencia_respiratoria_var = None
        self.prioridade_var = None
        self.priority_ctrl = None
        self.priority = None

    # Le o TXT
    def read_vital_signals(self, file_path):
        vital_signals = {}
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                victim_id = int(parts[0])
                signals = list(map(float, parts[1:]))
                vital_signals[victim_id] = signals
        return vital_signals
    
    def train_fuzzy_logic_system(self):

        # Inputs do sistema
        self.qPA_var = ctrl.Antecedent(np.arange(-10, 11, 1), 'qPA')
        self.pulso_var = ctrl.Antecedent(np.arange(0, 201, 1), 'pulso')
        self.frequencia_respiratoria_var = ctrl.Antecedent(np.arange(0, 21, 1), 'frequencia_respiratoria')

        # Output do sistema
        self.prioridade_var = ctrl.Consequent(np.arange(1, 201, 1), 'prioridade')

        # Dados das vítimas do TXT
        qPA_data = [signals[2] for signals in self.list_of_victims.values()]
        pulse_data = [signals[3] for signals in self.list_of_victims.values()]
        respiratory_data = [signals[4] for signals in self.list_of_victims.values()]

        # Divide os dados 
        qPA_quartiles = np.percentile(qPA_data, [20, 40, 75])
        pulse_quartiles = np.percentile(pulse_data, [25, 50, 75])
        respiratory_quartiles = np.percentile(respiratory_data, [25, 50, 75])

        self.qPA_var['Crítico'] = fuzz.trimf(self.qPA_var.universe, [np.min(qPA_data), qPA_quartiles[0], np.max(qPA_data)])
        self.qPA_var['Instável'] = fuzz.trimf(self.qPA_var.universe, [np.min(qPA_data), qPA_quartiles[0], qPA_quartiles[2]])
        self.qPA_var['Potencialmente Estável'] = fuzz.trimf(self.qPA_var.universe, [qPA_quartiles[0], qPA_quartiles[1], qPA_quartiles[2]])
        self.qPA_var['Estável'] = fuzz.trimf(self.qPA_var.universe, [qPA_quartiles[1], qPA_quartiles[2], np.max(qPA_data)])

        self.pulso_var['Crítico'] = fuzz.trimf(self.pulso_var.universe, [np.min(pulse_data), np.min(pulse_data), pulse_quartiles[0]])
        self.pulso_var['Instável'] = fuzz.trimf(self.pulso_var.universe, [np.min(pulse_data), pulse_quartiles[0], pulse_quartiles[1]])
        self.pulso_var['Potencialmente Estável'] = fuzz.trimf(self.pulso_var.universe, [pulse_quartiles[0], pulse_quartiles[1], pulse_quartiles[2]])
        self.pulso_var['Estável'] = fuzz.trimf(self.pulso_var.universe, [pulse_quartiles[1], pulse_quartiles[2], np.max(pulse_data)])

        self.frequencia_respiratoria_var['Crítico'] = fuzz.trimf(self.frequencia_respiratoria_var.universe, [np.min(respiratory_data), np.min(respiratory_data), respiratory_quartiles[0]])
        self.frequencia_respiratoria_var['Instável'] = fuzz.trimf(self.frequencia_respiratoria_var.universe, [np.min(respiratory_data), respiratory_quartiles[0], respiratory_quartiles[1]])
        self.frequencia_respiratoria_var['Potencialmente Estável'] = fuzz.trimf(self.frequencia_respiratoria_var.universe, [respiratory_quartiles[0], respiratory_quartiles[1], respiratory_quartiles[2]])
        self.frequencia_respiratoria_var['Estável'] = fuzz.trimf(self.frequencia_respiratoria_var.universe, [respiratory_quartiles[1], respiratory_quartiles[2], np.max(respiratory_data)])

        self.prioridade_var['Crítico'] = fuzz.trimf(self.prioridade_var.universe, [1, 25, 50])
        self.prioridade_var['Instável'] = fuzz.trimf(self.prioridade_var.universe, [50, 75, 100])
        self.prioridade_var['Potencialmente Estável'] = fuzz.trimf(self.prioridade_var.universe, [75, 100, 125])
        self.prioridade_var['Estável'] = fuzz.trimf(self.prioridade_var.universe, [100, 150, 200])

        rules = [
            ctrl.Rule(self.qPA_var['Crítico'] | self.pulso_var['Crítico'] | self.frequencia_respiratoria_var['Crítico'], self.prioridade_var['Crítico']),
            ctrl.Rule(self.qPA_var['Instável'] | self.pulso_var['Instável'] | self.frequencia_respiratoria_var['Instável'], self.prioridade_var['Instável']),
            ctrl.Rule(self.qPA_var['Potencialmente Estável'] & self.pulso_var['Potencialmente Estável'] & self.frequencia_respiratoria_var['Potencialmente Estável'], self.prioridade_var['Potencialmente Estável']),
            ctrl.Rule(self.qPA_var['Estável'] & self.pulso_var['Estável'] & self.frequencia_respiratoria_var['Estável'], self.prioridade_var['Estável'])
        ]

        self.priority_ctrl = ctrl.ControlSystem(rules)
        self.priority = ctrl.ControlSystemSimulation(self.priority_ctrl)
    
    def compute_priority(self, qPA, pulso, frequencia_respiratoria):
        self.priority.input['qPA'] = qPA
        self.priority.input['pulso'] = pulso
        self.priority.input['frequencia_respiratoria'] = frequencia_respiratoria

        self.priority.compute()

        return self.priority.output['prioridade'] 

if __name__ == '__main__':
    logic = FuzzyLogic()

    # Treina
    logic.train_fuzzy_logic_system()

    # Novo dado pra usar no sistema treinado
    priority = logic.compute_priority(-1, 100, 10)
    print("Priority:", priority)