import numpy as np


class Neuron:
    def __init__(self, num_inputs, scale=0.1):
        self.weights = np.random.normal(0, scale, num_inputs)
        self.bias = np.random.normal(0, scale)

    def get_sum(self, inputs):
        return np.dot(inputs, self.weights) + self.bias
