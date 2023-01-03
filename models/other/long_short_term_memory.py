import numpy as np
import pandas as pd
from numpy import exp
from numpy import tanh
from numpy.random import normal  # for weights initialization.


def sigmoid(x):
    return 1 / (1 + exp(-x))


def tanh_derivative(x):
    return 1 - x ** 2


def softmax(X):
    exponent = exp(X)
    total = np.sum(exponent, axis=1).reshape(-1, 1)
    result = exponent / total
    return result


class LSTM:

    def __init__(self):
        pass
