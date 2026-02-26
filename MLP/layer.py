import numpy as np
from .neuron import Neuron

class Layer:
    '''
    Represents a layer and performs forward/backward operations
    '''
    def __init__(self, num_inputs, num_neurons, activation_function):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]
        self.activation_function = activation_function
        self.inputs = np.zeros(num_inputs, dtype=float)
        self.outputs = np.zeros(num_neurons, dtype=float)

    def forward(self):
        '''
        Performs the forward pass calculations related to its layer

        Returns:
            np.array: the output produced by the layer during the forward pass
        '''
        weights = np.array([n.weights for n in self.neurons])       # (num_neurons, num_inputs)
        biases  = np.array([n.bias for n in self.neurons])          # (num_neurons,)
        v = weights.dot(self.inputs) + biases                       # (num_neurons, num_weights_per_neuron) * (num_inputs==num_weights_per_neuron, 1) + (num_biases==num_neurons)

        # Calculates final output of each neuron (one output per neuron)
        self.outputs = np.array([
            self.activation_function.calc_value(val) for val in v
        ])

        # Stores the outputs inside its respective neuron
        for neuron, out in zip(self.neurons, self.outputs):
            neuron.output = out

        return self.outputs

    def backward(self, next_layer):
        '''
        Performs the backwards pass calculations related to its layer
        (Calculates the deltas to be used in weight updating)
        Args:
            nex_layer: the layer object that proceeds the current layer
        '''
        # Gathers next-layer weights/deltas
        next_weights = np.array([n.weights for n in next_layer.neurons])
        next_deltas = np.array([n.delta for n in next_layer.neurons])

        for j, neuron in enumerate(self.neurons):
            error = np.dot(next_weights[:, j], next_deltas)
            neuron.delta = error * self.activation_function.calc_derivative(neuron.output)

    def update_weights_biases(self, eta, alpha):
        '''
        Updates weights and biases of all neurons in the layer
        (The biases are updated similarly to weights, however their input is always 1)
        Args:
            eta (learning rate): the learning rate hyper-parameter of the Network
            alpha (momentum): the momentum hyper-parameter of the Network
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
