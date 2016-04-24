import numpy as np


class NeuralNetwork(object):
    def __init__(self, func, deriv, *args, bias_vals=None, bias_weights=None):
        """
        Passed the list of weights of each layer and the evaluation function
        """

        self.weights = args
        self.bias_vals = bias_vals or [0 for weight in self.weights]
        self.bias_vals = np.array(self.bias_vals)
        self.bias_weights = bias_weights or [[0] * len(weight[0])
                                             for weight in self.weights]
        self.bias_weights = np.array(self.bias_weights)
        self.f = np.vectorize(func)
        self.f_prime = np.vectorize(deriv)
        self.num_inputs = len(args[0])

    def check_inputs(self, inputs):
        if len(inputs) != self.num_inputs:
            raise ValueError('Incorrect number of inputs')

    def calculate(self, inputs, intermediates=False):
        self.check_inputs(inputs)
        inputs = np.array(inputs)
        if intermediates:
            results = [inputs.tolist()]
        for bias_val, layer_weights, bias_weights in zip(
                self.bias_vals, self.weights, self.bias_weights):
            biased_inputs = np.hstack((inputs, bias_val))
            copy = layer_weights[:]
            copy.append(bias_weights)
            biased_weights = np.array(copy)
            inputs = self.f(biased_inputs.dot(biased_weights))
            if intermediates:
                results.append(inputs.tolist())

        if intermediates:
            return results
        else:
            return inputs.tolist()

    def train(self, inputs, output, learning_rate=0.9):
        neu_vals = np.array(self.calculate(inputs, True))
        new_weights = np.copy(self.weights)
        for i in range(len(neu_vals) - 1, -1, -1):
            if i is len(neu_vals) - 1:
                continue

            w_ij = np.array(self.weights[i])
            o_i = np.array(neu_vals[i])
            o_j = np.array(neu_vals[i + 1])

            if i is len(neu_vals) - 2:
                delta = (output - o_j) * self.f_prime(o_j)
            else:
                w_jk = np.array(self.weights[i + 1])
                delta = (deltas.dot(w_jk.transpose())) * (self.f_prime(o_j))

#   delta_weight = learning_rate * delta * o_i
            deltas = np.array(delta)
            new_weights[i] = ((o_i.reshape(
                len(o_i), -1) * delta * learning_rate) + w_ij).tolist()
        self.weights = np.array(new_weights)
        print(self.weights)
