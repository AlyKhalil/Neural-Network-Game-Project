import numpy as np
from .activationfunctions import Sigmoid, Linear
from .neuron import Neuron
from .layer import Layer

class Network:
    def __init__(self, num_inputs, num_outputs, num_hidden_neurons=2, eta=0.1, alpha=0.9):
        # Only ONE hidden layer and an output layer
        self.layers = [
            Layer(num_inputs, num_hidden_neurons, Sigmoid()),
            Layer(num_hidden_neurons, num_outputs, Linear())
        ]
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_neurons = num_hidden_neurons
        self.eta = eta
        self.alpha = alpha

    def get_weights_snapshot(self):
        '''
        Creates a deep copy of all the weights and biases in the Network

        Returns:
            list[list[tuples]]: snapshot containing all weights and biases in the Network,
            each list representing a Layer and each tuple a Neuron
        '''
        snapshot = []
        for layer in self.layers:
            layer_copy = []
            for neuron in layer.neurons:
                layer_copy.append((
                    neuron.weights.copy(),
                    float(neuron.bias)
                ))
            snapshot.append(layer_copy)
        return snapshot

    def set_weights_snapshot(self, snapshot):
        '''
        Sets all weights and biases provided by a snapshot to the respective Network instance

        Args:
            snapshot: list[list[tuple]] representing the weights and biases of the Network
        '''
        for L_idx, layer in enumerate(self.layers):
            for N_idx, neuron in enumerate(layer.neurons):
                weights_copy, bias_copy = snapshot[L_idx][N_idx]
                neuron.weights = weights_copy.copy()
                neuron.bias = float(bias_copy)

    def forward(self, inputs: np.array):
        '''
        Iterively calculates the output of each layer and passes it to the next layer
        And returns the final OUTPUT of the forward pass 
        Args:
            inputs: np.array of the inputs to the Network

        Returns:
            np.array: The final output/prediction of the Network
        '''
        for layer in self.layers:
            # set the inputs for the layer (no need to copy unless you want safety)
            layer.inputs = np.array(inputs, dtype=float)
            inputs = layer.forward()
        
        return inputs
       
    def backward(self, targets: np.array):
        '''
        Does ONE backwards pass over the Neural Network,
        for a single example, updating the weights and biases

        Args:
            targets: np.array of the desired outputs of the Network 
        '''
        output_layer = self.layers[-1]
        y = output_layer.outputs

        # Output layer delta
        for k, neuron in enumerate(output_layer.neurons):
            error = targets[k] - y[k]
            neuron.delta = error * output_layer.activation_function.calc_derivative(y[k])

        # Hidden layers deltas
        for i in reversed(range(len(self.layers) - 1)):
            self.layers[i].backward(self.layers[i + 1])

        # Weight updates
        for layer in self.layers:
            layer.update_weights_biases(self.eta, self.alpha)

    def train(self, training_examples: list[tuple]):
        '''
        Does ONE Epoch over the training data set
        that includes a forward and a backwards pass,
        updating the weights and biases, for each training example.
        (On-line Learning)
        Args:
        training_examples: list[tuples] of training examples each including both the inputs and the targets
        
        Returns:
            float: the RMS error of the whole Epoch 
        '''
        predictions = []
        targets = [] 
        for ex_inputs, ex_targets in training_examples:
            ex_inputs = np.array(ex_inputs, dtype=float)
            ex_targets = np.array(ex_targets, dtype=float)

            predictions.append(self.forward(ex_inputs))
            targets.append(ex_targets)
            self.backward(ex_targets)
        
        predictions = np.array(predictions)
        targets = np.array(targets)

        return np.sqrt(np.mean((targets - predictions) ** 2))

    def validate(self, validation_examples: list[tuple]):
        '''
        Performs a validation pass through the validation data set.
        (Forward pass only, no weight updating)
        Args:
            validation_examples: list[tuples] of validation examples each including both the inputs and the targets
        Returns:
            float: the RMS error of the validation pass
        '''
        predictions = []
        targets = []
        for ex_inputs, ex_targets in validation_examples:
            ex_inputs = np.array(ex_inputs, dtype=float)
            ex_targets = np.array(ex_targets, dtype=float)

            predictions.append(self.forward(ex_inputs))
            targets.append(ex_targets)
        
        predictions = np.array(predictions)
        targets = np.array(targets)

        return np.sqrt(np.mean((targets - predictions) ** 2))

    def test(self, test_examples: list[tuple]):
        '''
        Performs a test pass through the test data set.
        (Forward pass only, no weight updating)
        Args:
            test_examples: list[tuples] of test examples each including both the inputs and the targets
        Returns:
            float: the RMS error of the test pass
        '''
        predictions = []
        targets = []
        for ex_inputs, ex_targets in test_examples:
            ex_inputs = np.array(ex_inputs, dtype=float)
            ex_targets = np.array(ex_targets, dtype=float)

            predictions.append(self.forward(ex_inputs))
            targets.append(ex_targets)
        
        predictions = np.array(predictions)
        targets = np.array(targets)

        return np.sqrt(np.mean((targets - predictions) ** 2))