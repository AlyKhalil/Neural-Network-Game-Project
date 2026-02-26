import numpy as np
import random

class Neuron:
    '''
    Represents a single neuron, which stores its meta data
    '''
    def __init__(self, num_inputs):
        self.weights = np.array([random.uniform(-1.0, 1.0) for _ in range(num_inputs)], dtype=float)
        self.bias = random.uniform(-1.0,1.0)
        self.output = 0.0
        self.delta = 0.0
        self.prev_change_in_weights = np.zeros(num_inputs, dtype=float)
        self.prev_change_in_bias = 0.0
