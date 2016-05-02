from neuralnetwork import NeuralNetwork
import numpy as np
import math


class EmotionNetwork(object):

    file_name = 'emotion-network.p'
    num_inputs = 6

    @staticmethod
    def f(x):
        return 1 / (1 + math.e**(-x))

    @staticmethod
    def f_prime(x):
        return x * (1 - x)

    def __init__(self, load=False):

        if load is True:
            self.load()
        elif load is not False:
            self.load(load)
        else:
            # input_weights = np.random.normal(0, 1,
            #                                  (self.num_inputs, 1)).tolist()
            # input_weights = [[0.01] for i in range(280)]
            input_weights = [[0.3], [0.35], [0.6], [0.3], [0.2], [0.2]]

            self.nn = NeuralNetwork(self.f, self.f_prime, input_weights)

    def save(self, filename=file_name):
        self.nn.save(filename)

    def calculate(self, inputs, intermediates=False):
        return self.nn.calculate(inputs, intermediates)

    def train(self, inputs, output):
        return self.nn.train(inputs, output, learning_rate=0.5)

    def load(self, filename=file_name):
        self.nn = NeuralNetwork(self.f, self.f_prime, filename)
