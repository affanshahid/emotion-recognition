import numpy as np
import pickle


class NeuralNetwork(object):
    def __init__(self, func, deriv, *args, bias_vals=None, bias_weights=None):
        """
        Passed the list of weights of each layer and the evaluation function or a file name
        """

        if type(args[0]) is str:
            self.load(func, deriv, args[0])
            return None

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

    def train(self, inputs, output, learning_rate=0.03):
        neu_vals = np.array(self.calculate(inputs, True))
        new_weights = (np.copy(self.weights)).tolist()
        new_bias = (np.copy(self.bias_weights)).tolist()
        for i in range(len(neu_vals) - 1, -1, -1):
            if i is len(neu_vals) - 1:
                continue

            w_ij = np.array(self.weights[i])
            o_i = np.array(neu_vals[i])
            b_i = np.array([self.bias_vals[i]])
            w_bj = np.array(self.bias_weights[i])
            o_j = np.array(neu_vals[i + 1])
            if i is len(neu_vals) - 2:
                delta = (output - o_j) * self.f_prime(o_j)
            else:
                w_jk = np.array(self.weights[i + 1])
                delta = (deltas.dot(w_jk.transpose())) * (self.f_prime(o_j))

            #delta_weight = learning_rate * delta * o_i
            deltas = np.array(delta)
            new_weights[i] = ((o_i.reshape(
                len(o_i), -1) * delta * learning_rate) + w_ij).tolist()
            new_bias[i] = ((b_i.reshape(
                len(b_i), -1) * delta * learning_rate) + w_bj)[0].tolist()
        self.weights = new_weights
        self.bias_weights = new_bias
        return neu_vals

    def save(self, file_name):
        temp_f = self.f
        temp_f_prime = self.f_prime
        del self.f
        del self.f_prime
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)
        self.f = temp_f
        self.f_prime = temp_f_prime

    def load(self, func, func_prime, file_name):
        with open(file_name, 'rb') as file:
            load = pickle.load(file)
            self.weights = load.weights
            self.bias_vals = load.bias_vals
            self.bias_weights = load.bias_weights
            self.f = func
            self.f_prime = func_prime
            self.num_inputs = load.num_inputs
