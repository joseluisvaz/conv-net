import numpy as np

class Relu(object):
    @staticmethod
    def compute(self, x):
        return np.maximum(0, x)
    @staticmethod
    def deriv(self, x):
        return 1. * (x > 0)

class Sigmoid(object):
    def compute(self, x):
        return 1. / (1. + np.exp(-x))

    def deriv(self, x):
        y = self.compute(x)
        return y * (1. - y)

class MeanSquaredError(object):
    def compute(self, (X, Y)):
        return (1. / 2. * X.shape[0]) * ((X - Y) ** 2.)

    def deriv(self, (X, Y)):
        return (X - Y) / X.shape[0]

class CrossEntropy(object):
    @staticmethod
    def softmax(z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    @staticmethod
    def deriv(Y, T):
        return Y - T