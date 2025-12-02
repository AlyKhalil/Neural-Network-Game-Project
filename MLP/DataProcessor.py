import numpy as np
import json
import csv
import random
from .network import Network
from collections import defaultdict

class DataProcessor:
    def __init__(self, csv_path: str = ""):
        self.csv_path = csv_path
        self.data = None  # numpy array
        self.MN = None
        self.MX = None

    def read_file(self):
        """Reads a numeric CSV (no header) into a list of float lists using only the stdlib."""
        data = []
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                # if not row:
                #     continue
                floats = [float(x) for x in row]
                data.append(floats)

        self.data = np.array(data, dtype=float)
        return self.data

    def shuffle(self, seed=42): # seed=42 produces a deterministic shuffle
        random.seed(seed)
        n = len(self.data)
        indices = list(range(n)) # range(n) produces an iterable with elements from 0 to n-1

        # Fisher Yates algorithm
        for i in range(n - 1, 0, -1):
            j = random.randint(0, i)
            indices[i], indices[j] = indices[j], indices[i]
        
        self.data = self.data[indices]
        return self.data

    def split_data(self, train_ratio=0.7, val_ratio=0.15):
        n = len(self.data)

        train_end = int(train_ratio * n)
        val_end = train_end + int(val_ratio * n)

        train = self.data[:train_end]
        val = self.data[train_end:val_end]
        test = self.data[val_end:]

        return train, val, test

    def compute_min_max(self, train_array):
        self.MN = train_array.min(axis=0)
        self.MX = train_array.max(axis=0)

    def load_min_max(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        self.MN = np.array(data["MN"], dtype=float)
        self.MX = np.array(data["MX"], dtype=float)

    def normalize(self, arr):
        return (arr - self.MN) / (self.MX - self.MN)

    def denormalize_output(self, arr):
        return arr * (self.MX[-2:] - self.MN[-2:]) + self.MN[-2:]

    def save_csv(self, arr, filename):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            for row in arr:
                writer.writerow(row)

    def save_min_max(self, filename="min_max.json"):
        data = {"MN": self.MN.tolist(), "MX": self.MX.tolist()}
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

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

            if "param_type" in reader.fieldnames: # !!
                type_key = "param_type"
            elif "type" in reader.fieldnames:
                type_key = "type"
            else:
                raise ValueError("trained_weights.csv missing required column 'param_type' or 'type'")

            for row in reader:
                L = int(row["layer"])
                N = int(row["neuron"])
                idx = int(row["index"])
                val = float(row["value"])
                typ = row[type_key]

                # Creates a pointer to that position in the dictionary
                neuron = layers[L][N]

                if typ == "weight":
                    neuron["weights"][idx] = val
                elif typ == "bias":
                    neuron["bias"] = val
                else:
                    raise ValueError(f"Unknown param_type '{typ}' in weights file.")

        for L_idx, layer in enumerate(net.layers):
            for N_idx, neuron in enumerate(layer.neurons):
                entry = layers[L_idx][N_idx]
                weights = [entry["weights"][i] for i in sorted(entry["weights"].keys())]
                neuron.weights = np.array(weights, dtype=float)
                neuron.bias = float(entry["bias"])

        return net

#==========================================================

def process_data(file_path="ce889_dataCollection.csv"):
    # 1. Load CSV into numpy array
    processor = DataProcessor(file_path)
    data = processor.read_file()
    print(f"Data shape: {data.shape}")

    # 2. Shuffle rows
    processor.shuffle()
    print("Data shuffled.")

    # 3. Split into train/val/test
    train, val, test = processor.split_data()
    print("Split complete:")
    print("  Train:", train.shape)
    print("  Val:  ", val.shape)
    print("  Test: ", test.shape)

    # 4. Compute min/max from training data
    processor.compute_min_max(train)
    print("Computed min/max scaling values.")

    # 5. Normalize each split
    normalized_train = processor.normalize(train)
    normalized_val   = processor.normalize(val)
    normalized_test  = processor.normalize(test)

    # 6. Save normalized data to CSV
    processor.save_csv(normalized_train, "normalized_training_data_ce889_dataCollection.csv")
    processor.save_csv(normalized_val,   "normalized_validation_data_ce889_dataCollection.csv")
    processor.save_csv(normalized_test,  "normalized_test_data_ce889_dataCollection.csv")
    print("Normalized datasets saved.")

    # 7. Save scaling values for inference use
    processor.save_min_max("scaling_values.json")
    print("Min/max scaling saved to scaling_values.json")

    # 8. Reload to verify the values
    processor.load_min_max("scaling_values.json")
    print("\nReloaded MN and MX:")
    print("MN:", processor.MN)
    print("MX:", processor.MX)

    print("\n=== Preprocessing Pipeline Complete ===")
