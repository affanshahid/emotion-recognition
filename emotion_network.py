from neuralnetwork import NeuralNetwork
import numpy as np
import math


class EmotionNetwork(object):

    file_name = 'emotion-network.p'
    pixels = 280

    @staticmethod
    def f(x):
        return 1 / (1 + math.e**(-x))

    @staticmethod
    def f_prime(x):
        return x * (1 - x)

    def __init__(self, load=False):

        if load:
            self.load()
        else:
            # input_weights = np.random.normal(0, 1, (self.pixels, 1)).tolist()
            input_weights = [[0.01] for i in range(280)]
            self.nn = NeuralNetwork(self.f, self.f_prime, input_weights)

    def save(self):
        self.nn.save(self.file_name)

    def calculate(self, inputs, intermediates=False):
        return self.nn.calculate(inputs, intermediates)

    def train(self, inputs, output):
        return self.nn.train(inputs, output, learning_rate=0.01)

    def load(self):
        return self.nn = NeuralNetwork(self.f, self.f_prime, self.file_name)
