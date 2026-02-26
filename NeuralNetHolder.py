import numpy as np
import json
from MLP import DataProcessor
from MLP import Network


class NeuralNetHolder:
    def __init__(self,
        weights_csv="trained_weights.csv",
        params_json="best_parameters.json",
        minmax_json="scaling_values.json"):


        self.processor = DataProcessor()
        self.net = self.processor.load_trained_network(
            weights_csv=weights_csv,
            params_json=params_json,
            minmax_json=minmax_json
        )
        print("Neural network loaded with parameters: ")
        print("hidden neurons:", self.net.num_hidden_neurons)
        print("Learning rate (eta):", self.net.eta)
        print("Momentum (alpha):", self.net.alpha)


    def predict(self, input_row: str):
        raw = [float(v.strip()) for v in input_row.split(",")]

        inp = np.array(raw, dtype=float)
        MN = self.processor.MN[:2]
        MX = self.processor.MX[:2]
        normalized = (inp - MN) / (MX - MN)


        out = self.net.forward(normalized)
        out = self.processor.denormalize_output(np.array(out))


        return out[1], out[0]