#---test_network.py---
import unittest
import numpy as np

from .activationfunctions import Sigmoid, Linear
from .neuron import Neuron
from .layer import Layer
from .network import Network


# -------------------------------------------------
#   Activation Function Tests
# -------------------------------------------------
class TestActivationFunctions(unittest.TestCase):

    def test_sigmoid_value(self):
        sig = Sigmoid()
        self.assertAlmostEqual(sig.calc_value(0), 0.5)

    def test_sigmoid_derivative(self):
        sig = Sigmoid()
        y = sig.calc_value(0)
        self.assertAlmostEqual(sig.calc_derivative(y), 0.25, places=5)


# -------------------------------------------------
#   Neuron Tests
# -------------------------------------------------
class TestNeuron(unittest.TestCase):

    def test_neuron_initialization(self):
        neuron = Neuron(3)
        self.assertEqual(len(neuron.weights), 3)
        self.assertTrue(-1.0 <= neuron.bias <= 1.0)


# -------------------------------------------------
#   Layer Tests
# -------------------------------------------------
class TestLayer(unittest.TestCase):

    def test_forward(self):
        layer = Layer(2, 2, Sigmoid())
        layer.inputs = np.array([1.0, 1.0])
        output = layer.forward()
        self.assertEqual(len(output), 2)
        self.assertTrue(np.all(output >= 0))
        self.assertTrue(np.all(output <= 1))


# -------------------------------------------------
#   Network Tests
# -------------------------------------------------
class TestNetwork(unittest.TestCase):

    def test_forward_pass_shapes(self):
        net = Network([
            Layer(2, 2, Sigmoid()),
            Layer(2, 1, Sigmoid())
        ])
        x = np.array([0.5, 0.5])
        y = net.forward(x)

        self.assertEqual(y.shape, (1,))
        self.assertTrue(0 <= y[0] <= 1)

    def test_backprop_runs(self):
        net = Network([
            Layer(2, 2, Sigmoid()),
            Layer(2, 1, Sigmoid())
        ])

        inputs = np.array([0, 1])
        targets = np.array([1])

        net.forward(inputs)
        error = net.backward(targets, eta=0.5)

        self.assertGreaterEqual(error, 0)
        self.assertIsNotNone(error)

    def test_xor_learning(self):
        training_data = [
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [0])
        ]

        net = Network([
            Layer(2, 2, Sigmoid()),
            Layer(2, 1, Sigmoid())
        ])

        # Train long enough to guarantee convergence
        net.train(training_data, eta=0.5, alpha=0.5, epochs=1000)

        # Test XOR results
        results = []
        for x, y in training_data:
            pred = net.forward(np.array(x))[0]
            results.append((x, pred, y[0]))

        # XOR correctness: prediction close to target
        for x, pred, target in results:
            if target == 1:
                self.assertGreater(pred, 0.8)  # Should be near 1
            else:
                self.assertLess(pred, 0.2)     # Should be near 0


if __name__ == '__main__':
    unittest.main()
