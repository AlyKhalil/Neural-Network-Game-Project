#---Layer.py---
import numpy as np
from .neuron import Neuron

class Layer:
    '''Represents a layer and performs forward/backward operations'''

    def __init__(self, num_inputs, num_neurons, activation_function):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]
        self.activation_function = activation_function
        self.inputs = np.zeros(num_inputs, dtype=float)
        self.outputs = np.zeros(num_neurons, dtype=float)

        # used inside forward/backward/update methods to avoid repeated allocations
        self._weights_mat = None
        self._biases = None
        self._deltas = np.zeros(num_neurons, dtype=float)

    def forward(self):
        weights = np.array([n.weights for n in self.neurons])       # (num_neurons, num_inputs)
        biases  = np.array([n.bias for n in self.neurons])          # (num_neurons,)
        v = weights.dot(self.inputs) + biases                       # pre-activations

        self.outputs = np.array([
            self.activation_function.calc_value(val) for val in v
        ])

        for neuron, out in zip(self.neurons, self.outputs):
            neuron.output = out

        return self.outputs

        # self.outputs = []
        # for neuron in self.neurons:
        #     v = np.dot(neuron.weights, self.inputs) + neuron.bias
        #     neuron.output = self.activation_function.calc_value(v)
        #     self.outputs.append(neuron.output)

        # self.outputs = np.array(self.outputs)
        # return self.outputs

    def backward(self, next_layer):
        # Gather next-layer weights/deltas
        next_weights = np.array([n.weights for n in next_layer.neurons])
        next_deltas = np.array([n.delta for n in next_layer.neurons])

        # Loop neurons normally, but compute error vectorized
        for j, neuron in enumerate(self.neurons):
            error = np.dot(next_weights[:, j], next_deltas)
            neuron.delta = error * self.activation_function.calc_derivative(neuron.output)

        # for j, neuron in enumerate(self.neurons):
        #     error = sum(next_neuron.weights[j] * next_neuron.delta
        #                 for next_neuron in next_layer.neurons)
        #     neuron.delta = error * self.activation_function.calc_derivative(neuron.output)

    def update_weights_biases(self, eta, alpha):
        '''
        Updates weights and biases of all neurons in the layer#
        (The biases are updated similarly to weights, however their input is always 1)
        '''
        for neuron in self.neurons:
            change_in_weights = (
                eta * neuron.delta * self.inputs + 
                alpha * neuron.prev_change_in_weights
            )

            change_in_bias = (
                eta * neuron.delta + 
                alpha * neuron.prev_change_in_bias
            )
            
            neuron.weights += change_in_weights
            neuron.bias += change_in_bias

            neuron.prev_change_in_weights = change_in_weights
            neuron.prev_change_in_bias = change_in_bias
