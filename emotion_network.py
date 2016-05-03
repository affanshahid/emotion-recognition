from neuralnetwork import NeuralNetwork
import numpy as np
import math
import random


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
        # inputs: reyeh, leyeh, mw, mh, ntr, ntl
        # hidden = 6
        # outputs: happy, sad, surprise

        if load is True:
            self.load()
        elif load is not False:
            self.load(load)
        else:
            # working weights for happy/neutral
            # input_weights = [[0.3], [0.35], [0.6], [0.3], [0.2], [0.2]]

            inputs_num = 6
            hidden_num = 7
            outputs_num = 3

            # inputs_range = ((-1 / math.sqrt(inputs_num)),
            #                 (1 / math.sqrt(inputs_num)))

            # input_weights = [[random.uniform(*inputs_range)
            #                   for _ in range(hidden_num)]
            #                  for _ in range(inputs_num)]

            # hidden_range = ((-1 / math.sqrt(hidden_num)),
            #                 (1 / math.sqrt(hidden_num)))
            # hidden_weights = [[random.uniform(*hidden_range)
            #                    for _ in range(outputs_num)]
            #                   for _ in range(hidden_num)]

            input_weights = [
                [0.3] * hidden_num, [0.3] * hidden_num, [0.4] * hidden_num,
                [0.5] * hidden_num, [0.35] * hidden_num, [0.35] * hidden_num
            ]

            hidden_weights = [
                [0.3] * outputs_num, [0.3] * outputs_num, [0.4] * outputs_num,
                [0.5] * outputs_num, [0.35] * outputs_num,
                [0.35] * outputs_num, [0.6] * outputs_num
            ]

            self.nn = NeuralNetwork(self.f,
                                    self.f_prime,
                                    input_weights,
                                    hidden_weights,
                                    bias_vals=[0, 0])

    def save(self, filename=file_name):
        self.nn.save(filename)

    def calculate(self, inputs, intermediates=False):
        return self.nn.calculate(inputs, intermediates)

    def train(self, inputs, output):
        return self.nn.train(inputs, output, learning_rate=0.5)

    def load(self, filename=file_name):
        self.nn = NeuralNetwork(self.f, self.f_prime, filename)

    def predict(self, input):
        outputs = self.calculate(input)
        max = 0
        for i, val in enumerate(outputs):
            if val > outputs[max]:
                max = i

        # if outputs[i] < 0.5:
        #     return '*UNCERTAIN*'

        if max is 0:
            return 'happy'
        elif max is 1:
            return 'neutral'
        elif max is 2:
            return 'surprise'
