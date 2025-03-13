import numpy as np
import math
import pickle

class Neuron:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.weights = [np.random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = np.random.uniform(-1, 1)
        self.output = 0
        self.inputs = []
        self.error_gradient = 0

class Layer:
    def __init__(self, num_neurons, num_inputs_per_neuron):
        self.num_neurons = num_neurons
        self.neurons = [Neuron(num_inputs_per_neuron) for _ in range(num_neurons)]

class ANN:
    def __init__(self, num_inputs, num_outputs, num_hidden, num_neurons_per_hidden, alpha):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden = num_hidden
        self.num_neurons_per_hidden = num_neurons_per_hidden
        self.alpha = alpha
        self.layers = []
        
        # Criar a arquitetura da rede
        if num_hidden > 0:
            # Primeira camada oculta
            self.layers.append(Layer(num_neurons_per_hidden, num_inputs))
            
            # Camadas ocultas adicionais
            for i in range(num_hidden - 1):
                self.layers.append(Layer(num_neurons_per_hidden, num_neurons_per_hidden))
            
            # Camada de saída
            self.layers.append(Layer(num_outputs, num_neurons_per_hidden))
        else:
            # Sem camadas ocultas, apenas entrada -> saída
            self.layers.append(Layer(num_outputs, num_inputs))
    
    def train(self, input_values, desired_output):

        output_values = self.calculate_output(input_values, desired_output)
        self.update_weights(output_values, desired_output)
        return output_values
    
    def calculate_output(self, input_values, desired_output=None):

        if desired_output is None:
            # Criar uma lista de zeros se não for fornecida uma saída desejada
            desired_output = [0] * self.num_outputs
            
        inputs = []
        output_values = []
        current_input = 0
        

        if len(input_values) != self.num_inputs:
            print(f"ERRO: O número de entradas deve ser {self.num_inputs}")
            return output_values
        
        inputs = input_values.copy()
        
        # Propagar a entrada pela rede (forward pass)
        for i in range(self.num_hidden + 1):
            if i > 0:
                inputs = output_values.copy()
            output_values = []
            
            for j in range(self.layers[i].num_neurons):
                N = 0
                self.layers[i].neurons[j].inputs = []
                
                for k in range(self.layers[i].neurons[j].num_inputs):
                    self.layers[i].neurons[j].inputs.append(inputs[current_input])
                    N += self.layers[i].neurons[j].weights[k] * inputs[current_input]
                    current_input += 1
                
                N -= self.layers[i].neurons[j].bias
                
                # Limitar N para evitar overflow (importante!)
                # N = max(-100, min(100, N))
                
                # Aplicar função de ativação
                if i == self.num_hidden:
                    self.layers[i].neurons[j].output = self.activation_function_output(N)
                else:
                    self.layers[i].neurons[j].output = self.activation_function(N)
                
                output_values.append(self.layers[i].neurons[j].output)
                current_input = 0
        
        return output_values
    
    def update_weights(self, outputs, desired_output):
        """Atualiza os pesos da rede usando backpropagation."""
        # Começar pela última camada e ir para trás
        for i in range(self.num_hidden, -1, -1):
            for j in range(self.layers[i].num_neurons):
                # Calcular o gradiente de erro
                if i == self.num_hidden:
                    # Camada de saída
                    error = desired_output[j] - outputs[j]
                    self.layers[i].neurons[j].error_gradient = outputs[j] * (1 - outputs[j]) * error
                else:
                    # Camadas ocultas
                    self.layers[i].neurons[j].error_gradient = self.layers[i].neurons[j].output * (1 - self.layers[i].neurons[j].output)
                    error_grad_sum = 0
                    for p in range(self.layers[i+1].num_neurons):
                        error_grad_sum += self.layers[i+1].neurons[p].error_gradient * self.layers[i+1].neurons[p].weights[j]
                    self.layers[i].neurons[j].error_gradient *= error_grad_sum
                
                # Limitar o gradiente para evitar explosão de gradientes
                self.layers[i].neurons[j].error_gradient = max(-1.0, min(1.0, self.layers[i].neurons[j].error_gradient))
                
                # Atualizar pesos
                for k in range(self.layers[i].neurons[j].num_inputs):
                    if i == self.num_hidden:
                        error = desired_output[j] - outputs[j]
                        delta = self.alpha * self.layers[i].neurons[j].inputs[k] * error
                    else:
                        delta = self.alpha * self.layers[i].neurons[j].inputs[k] * self.layers[i].neurons[j].error_gradient
                    
                    self.layers[i].neurons[j].weights[k] += delta
                
                # Atualizar bias
                bias_delta = self.alpha * -1 * self.layers[i].neurons[j].error_gradient
                self.layers[i].neurons[j].bias += bias_delta
    
    def activation_function(self, value):
        return self.tanh(value)
    
    def activation_function_output(self, value):
        return self.tanh(value)
    
    # def tanh(self, value):
    #     """Função de ativação tangente hiperbólica  """
    #     k = math.exp(-2 * value)
    #     return 2 / (1.0 + k) - 1
    
    def tanh(self, value):
        """Função de ativação tangente hiperbólica  """
        try:
            k = math.exp(-2 * value)
            return 2 / (1.0 + k) - 1
        except OverflowError:
            return -1.0 if value < 0 else 1.0
    def sigmoid(self, value):
        """Função de ativação sigmoide """
        k = math.exp(value)
        return k / (1.0 + k)

    def save_weights(self, filename):
        """Guarda os pesos da rede num ficheiro de texto."""
        weight_str = ""
        
        # Percorre todas as camadas e neurónios
        for layer in self.layers:
            for neuron in layer.neurons:
                # Guarda os pesos de cada neurónio
                for weight in neuron.weights:
                    weight_str += str(weight) + ","
                # Guarda também o bias
                weight_str += str(neuron.bias) + ","
        

        if weight_str:
            weight_str = weight_str[:-1]
        

        with open(filename, 'w') as f:
            f.write(weight_str)
        
        return "Modelo guardado com sucesso em " + filename
    
    def load_weights(self, filename):
        """Carrega os pesos da rede a partir de um ficheiro de texto."""
        try:
            with open(filename, 'r') as f:
                weight_str = f.read()
            
            if not weight_str:
                return "Ficheiro vazio."
            
            weight_values = weight_str.split(',')
            w = 0   
            
            # Aplica os pesos carregados à rede
            for layer in self.layers:
                for neuron in layer.neurons:
                    for i in range(len(neuron.weights)):
                        neuron.weights[i] = float(weight_values[w])
                        w += 1
                    neuron.bias = float(weight_values[w])
                    w += 1
            
            return "Modelo carregado com sucesso de " + filename
        except Exception as e:
            return f"Erro ao carregar modelo: {str(e)}"
    
 