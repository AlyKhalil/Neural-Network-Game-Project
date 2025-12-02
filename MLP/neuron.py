#---Neuron.py---
import numpy as np
import random

class Neuron:
    '''Represents a single neuron and stores its parameters'''

    def __init__(self, num_inputs):
        self.weights = np.array([random.uniform(-1.0, 1.0) for _ in range(num_inputs)])
        # self.weights = np.random.uniform(-1.0, 1.0, num_inputs) # CHANGE
        self.bias = random.uniform(-1.0,1.0)
        # self.bias = np.random.uniform(-1.0, 1.0) # CHANGE
        self.output = 0.0
        self.delta = 0.0
        self.prev_change_in_weights = np.zeros(num_inputs)
        self.prev_change_in_bias = 0.0
