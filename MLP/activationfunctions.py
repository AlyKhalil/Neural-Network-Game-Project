import numpy as np

class Sigmoid:
    def __init__(self, lmbda=1.0):
        self.lmbda = lmbda
    
    def calc_value(self, v):
        return 1 / (1 + np.exp(-self.lmbda * v))
    
    def calc_derivative(self, y):
        # y = sigmoid(v), so derivative = lambda * y(1 - y)
        return self.lmbda * y * (1 - y)


class ReLU:
    def calc_value(self, v):
        return max(0, v)

    def calc_derivative(self, y):
        return 1.0 if y > 0 else 0.0


class Linear:
    def calc_value(self, v):
        return v
    
    def calc_derivative(self, y):
        return 1.0
