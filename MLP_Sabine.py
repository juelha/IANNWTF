import numpy as np


#### Preparation ####

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidprime(x):
    return sigmoid(x) * (1 - sigmoid(x))


#### Data ####
# 1. possible inputs are 1 and 0

input_pairs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

labels = {
'and': [False, False, False, True],
'or': [False, True, True, True],
'nand': [True, True, True, False],
'nor': [True, False, False, False],
'xor': [False, True, True, False]}


class Perceptron():

    def __init__(self, input_units, learning_rate=1):
        self.weights = np.random.randn(1, self.input_units)  # np 1D  array the length of the input units
        self.bias = np.random.randn(1, 1)
        self.alpha = learning_rate

    def forward_step(self, inputs):
        '''
        :param inputs: an np array of input values
        computes the net input, weights and biases (drive) and outputs the activation
        :return: float of the activation
        '''

        net_input = (inputs @ self.weights) + self.bias  # product of all inputs and their weights + bias
        sigmoid_activation = 1 / (1 + np.exp(-net_input))
        return sigmoid_activation

    def update(self, delta):
        # how to access the last activation value for the weight in question?
        gradient = delta * self.forward_step()

        self.weights -= self.alpha * gradient
