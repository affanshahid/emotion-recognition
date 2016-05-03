import sys
import math
sys.path.append('..')
from neuralnetwork import NeuralNetwork


def main():
    def f(x):
        return 1 / (1 + math.e**(-x))

    def f_prime(x):
        return x * (1 - x)

    net = NeuralNetwork(f,
                        f_prime,
                        [[0.2, -0.3], [0.4, 0.1], [-0.5, 0.2]],
                        [[-0.3], [-0.2]],
                        bias_vals=[1, 1],
                        bias_weights=[[-0.4, 0.2], [0.1]])

    net = NeuralNetwork(f,
                        f_prime,
                        [[0.1, 0, 0.3], [-0.20, 0.2, -0.4]],
                        [[-0.4, 0.2], [0.1, -0.1], [0.6, -0.2]],
                        bias_vals=[1, 1],
                        bias_weights=[[0.1, 0.2, 0.5], [-0.1, 0.6]])
    print(net.weights, 'init')
    net.train([0.6, 0.1], [1, 0], learning_rate=0.1)
    net.train([0.6, 0.1], [1, 0], learning_rate=0.1)
    # net.train([1, 0, 1], [1], learning_rate=0.9)

    print(net.weights)


if __name__ == '__main__':
    sys.exit(int(main() or 0))
