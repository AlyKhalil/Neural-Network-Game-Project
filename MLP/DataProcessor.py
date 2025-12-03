# --- DataProcessor.py ---
import numpy as np
import json
import csv
import random
from collections import defaultdict
from .network import Network


class DataProcessor:
    def __init__(self, csv_path: str = ""):
        self.csv_path = csv_path
        self.data = None
        self.MN = None
        self.MX = None

    # ---------------------------------------------------------
    # STATIC CSV I/O (used everywhere, including util.py)
    # ---------------------------------------------------------
    @staticmethod
    def load_csv(path):
        """Static CSV loader: reads a numeric CSV without headers."""
        data = []
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                floats = [float(x) for x in row]
                data.append(floats)
        return np.array(data, dtype=float)

    @staticmethod
    def save_csv(arr, path):
        """Static CSV writer."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            for row in arr:
                writer.writerow(row)

    # ---------------------------------------------------------
    # Shuffle (index-based unified implementation)
    # ---------------------------------------------------------
    @staticmethod
    def shuffle_indices(n, seed=None):
        """Return a shuffled index array using Fisher–Yates.
        If seed is None, use system randomness; otherwise use provided seed (int)."""
        if seed is None:
            # system randomness
            # use random.SystemRandom for cryptographic-quality shuffle source
            sysrand = random.SystemRandom()
            indices = list(range(n))
            for i in range(n - 1, 0, -1):
                j = sysrand.randint(0, i)
                indices[i], indices[j] = indices[j], indices[i]
            return np.array(indices, dtype=int)
        else:
            random.seed(seed)
            indices = list(range(n))
            for i in range(n - 1, 0, -1):
                j = random.randint(0, i)
                indices[i], indices[j] = indices[j], indices[i]
            return np.array(indices, dtype=int)

    # ---------------------------------------------------------
    # Data splitting
    # ---------------------------------------------------------
    def split_data(self, train_ratio=0.7, val_ratio=0.15):
        if self.data is None:
            raise ValueError("DataProcessor.split_data: processor.data is None")
        n = len(self.data)
        train_end = int(train_ratio * n)
        val_end = train_end + int(val_ratio * n)
        train = self.data[:train_end]
        val = self.data[train_end:val_end]
        test = self.data[val_end:]
        return train, val, test

    # ---------------------------------------------------------
    # Scaling
    # ---------------------------------------------------------
    def compute_min_max(self, train_array):
        self.MN = train_array.min(axis=0)
        self.MX = train_array.max(axis=0)

    def load_min_max(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        self.MN = np.array(data["MN"], dtype=float)
        self.MX = np.array(data["MX"], dtype=float)

    def save_min_max(self, filename="min_max.json"):
        data = {"MN": self.MN.tolist(), "MX": self.MX.tolist()}
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def normalize(self, arr):
        return (arr - self.MN) / (self.MX - self.MN)

    def denormalize_output(self, arr):
        # NOTE: assumes outputs are the last 2 columns - keep or generalize
        return arr * (self.MX[-2:] - self.MN[-2:]) + self.MN[-2:]

    # ---------------------------------------------------------
    # Load trained network
    # ---------------------------------------------------------
    def load_trained_network(self, weights_csv, params_json, minmax_json='scaling_values.json'):
        self.load_min_max(minmax_json)

        with open(params_json, "r") as f:
            params = json.load(f)

        hidden = int(params["hidden_neurons"])
        eta = float(params["eta"])
        alpha = float(params["alpha"])

        net = Network(
            num_inputs=2,
            num_outputs=2,
            num_hidden_neurons=hidden,
            eta=eta,
            alpha=alpha
        )

        layers = defaultdict(lambda: defaultdict(lambda: {"weights": {}, "bias": None}))

        with open(weights_csv, newline="") as f:
            reader = csv.DictReader(f)

            if "param_type" in reader.fieldnames:
                type_key = "param_type"
            elif "type" in reader.fieldnames:
                type_key = "type"
            else:
                raise ValueError("trained_weights.csv missing 'param_type' or 'type'")

            for row in reader:
                L = int(row["layer"])
                N = int(row["neuron"])
                idx = int(row["index"])
                val = float(row["value"])
                typ = row[type_key]

                neuron = layers[L][N]

                if typ == "weight":
                    neuron["weights"][idx] = val
                elif typ == "bias":
                    neuron["bias"] = val
                else:
                    raise ValueError(f"Unknown param_type '{typ}'.")

        # Attach weights to network instance
        for L_idx, layer in enumerate(net.layers):
            for N_idx, neuron in enumerate(layer.neurons):
                entry = layers[L_idx][N_idx]
                weights = [entry["weights"][i] for i in sorted(entry["weights"].keys())]
                neuron.weights = np.array(weights, dtype=float)
                neuron.bias = float(entry["bias"])

        return net


# ==========================================================
def process_data(file_path="ce889_dataCollection.csv"):
    processor = DataProcessor(file_path)

    # Load file via static method
    data = DataProcessor.load_csv(file_path)
    processor.data = data
    print(f"Data shape: {data.shape}")

    # Shuffle via unified index shuffler
    perm = DataProcessor.shuffle_indices(len(data))
    data = data[perm]
    processor.data = data
    print("Data shuffled.")

    # Split
    train, val, test = processor.split_data()
    print("Split complete:")
    print("  Train:", train.shape)
    print("  Val:  ", val.shape)
    print("  Test: ", test.shape)

    # Scaling
    processor.compute_min_max(train)
    print("Computed min/max.")

    norm_train = processor.normalize(train)
    norm_val = processor.normalize(val)
    norm_test = processor.normalize(test)

    # Save normalized datasets
    DataProcessor.save_csv(norm_train, "normalized_training_data_ce889_dataCollection.csv")
    DataProcessor.save_csv(norm_val,   "normalized_validation_data_ce889_dataCollection.csv")
    DataProcessor.save_csv(norm_test,  "normalized_test_data_ce889_dataCollection.csv")
    print("Saved normalized datasets.")

    # Save min/max scaling
    processor.save_min_max("scaling_values.json")
    print("Saved scaling_values.json")

    # Reload (optional)
    processor.load_min_max("scaling_values.json")
    print("\nReloaded MN:", processor.MN)
    print("Reloaded MX:", processor.MX)

    print("\n=== Preprocessing Complete ===")
