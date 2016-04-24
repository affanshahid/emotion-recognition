from neuralnetwork import NeuralNetwork
import numpy as np
import math


class EmotionNetwork(object):

    file_name = 'emotion-network.p'
    pixels = 280

    def f(x):
        return 1 / (1 + math.e**(-x))

    def f_prime(x):
        return x * (1 - x)

    def __init__(self, load=False):

        if load:
            self.nn = NeuralNetwork(self.f, self.f_prime, self.file_name)
        else:
            input_weights = np.random.normal(0, 1, (self.pixels, 1)).tolist()
            self.nn = NeuralNetwork(self.f, self.f_prime, input_weights)

    def save(self):
        self.nn.save(self.file_name)

    def calculate(self, inputs):
        return self.nn.calculate(inputs)
