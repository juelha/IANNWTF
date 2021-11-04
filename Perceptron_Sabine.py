import numpy as np

class Perceptron(input_units):
    def __init__(self, input_units):
        self.labels = 'labels'
        self.weights = np.random.randn(1, self.input_units)
        self.bias = np.random.randn(1, 1)
        self.alpha = 1

    def forward_step(self, inputs):
        net_input = np.dot(inputs, self.weights) + self.bias # product of all inputs and their weights + bias
        activation = 1/(1 + np.exp(net_input))
        return activation

    def update(self, delta):
        gradient = delta * activation




