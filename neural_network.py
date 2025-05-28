import numpy as np
import random

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_ih = np.random.randn(self.hidden_size, self.input_size) * np.sqrt(2. / self.input_size)
        self.weights_ho = np.random.randn(self.output_size, self.hidden_size) * np.sqrt(2. / self.hidden_size)
        self.bias_h = np.zeros((self.hidden_size, 1))
        self.bias_o = np.zeros((self.output_size, 1))

    def feedforward(self, inputs):
        hidden = np.dot(self.weights_ih, inputs) + self.bias_h
        hidden = np.maximum(0, hidden)  # ReLU activation
        output = np.dot(self.weights_ho, hidden) + self.bias_o
        # Clip the values to avoid overflow in the sigmoid function
        output = np.clip(output, -500, 500)
        output = 1 / (1 + np.exp(-output))  # Sigmoid activation
        return output

    def mutate(self, mutation_rate):
        def mutate_value(value):
            if random.random() < mutation_rate:
                return value + np.random.normal() * 0.5
            return value

        vectorize_mutate = np.vectorize(mutate_value)
        self.weights_ih = vectorize_mutate(self.weights_ih)
        self.weights_ho = vectorize_mutate(self.weights_ho)
        self.bias_h = vectorize_mutate(self.bias_h)
        self.bias_o = vectorize_mutate(self.bias_o)

def crossover(parent1, parent2):
    child = NeuralNetwork(parent1.input_size, parent1.hidden_size, parent1.output_size)
    for attr in ['weights_ih', 'weights_ho', 'bias_h', 'bias_o']:
        parent1_attr = getattr(parent1, attr)
        parent2_attr = getattr(parent2, attr)
        mask = np.random.rand(*parent1_attr.shape) > 0.5
        setattr(child, attr, np.where(mask, parent1_attr, parent2_attr))
    return child

def create_population(size, input_size, hidden_size, output_size):
    return [NeuralNetwork(input_size, hidden_size, output_size) for _ in range(size)]
