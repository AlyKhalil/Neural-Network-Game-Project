import numpy as np
import json
import csv
import random
from collections import defaultdict
from .network import Network


class DataProcessor:
    '''
    Class responsible for Data Manipulation
    '''
    def __init__(self, csv_path: str = ""):
        self.csv_path = csv_path
        self.data = None
        self.MN = None
        self.MX = None

    @staticmethod
    def load_csv(path):
        '''
        Loads Numeric CSV files into a numpy array
        Args:
            path: path to the CSV file
        '''
        # NOTE: assumes all data is numeric and it does not automatically set self.data
        # self.data must be set manually after loading if necessary.
        data = []
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                data.append([float(x) for x in row])
        return np.array(data, dtype=float)

    @staticmethod
    def save_csv(arr, filename):
        '''
        Saves a numpy array to a CSV file
        Args:
            arr: numpy array to save
            filename: name of the CSV file
        '''
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            for row in arr:
                writer.writerow(row)

    @staticmethod
    def shuffle_indices(n, seed=None):
        '''
        Takes the length of an array and shuffles the indicies using Fisher-Yates.
        If seed is None, use system randomness, for true randomness. A seed can be provided
        for reproducible shuffling.
        Args:
            n: length of the array to shuffle
            seed (optional): seed for random generator. Defaults to None.
        
        Returns:
            np.array: array of shuffled indices
        '''
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

    def split_data(self, train_ratio=0.7, val_ratio=0.15):
        '''
        Splits self.data into train, validation, and test sets
        according to the provided ratios.
        Args:
            train_ratio: ratio of data to use for training, defaults to 0.7
            val_ratio: ratio of data to use for validation, defaults to 0.15
        
        Returns:
            tuple: (train_set, val_set, test_set)
        '''
        if self.data is None:
            raise ValueError("DataProcessor.split_data: processor.data is None")
        n = len(self.data)
        train_end = int(train_ratio * n)
        val_end = train_end + int(val_ratio * n)
        train = self.data[:train_end]
        val = self.data[train_end:val_end]
        test = self.data[val_end:]
        return train, val, test
        
    def compute_min_max(self, train_array):
        '''
        Finds min and max values for all columns in the training data
        and stores them in self.MN and self.MX
        Args:
            train_array: numpy array of training data
        '''
        self.MN = train_array.min(axis=0)
        self.MX = train_array.max(axis=0)

    def load_min_max(self, filename):
        '''
        Loads min and max values into self.MN and self.MX from a JSON file
        Args:
            filename: path to JSON file
        '''
        with open(filename, "r") as f:
            data = json.load(f)
        self.MN = np.array(data["MN"], dtype=float)
        self.MX = np.array(data["MX"], dtype=float)

    def save_min_max(self, filename="min_max.json"):
        '''
        Saves self.MN and self.MX to a JSON file
        Args:
            filename: name of JSON file
        '''
        data = {"MN": self.MN.tolist(), "MX": self.MX.tolist()}
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def normalize(self, arr):
        '''
        Normalizes the input array using self.MN and self.MX
        Args:
            arr: numpy array to normalize
        
        Returns:
            numpy array: normalized array
        '''
        return (arr - self.MN) / (self.MX - self.MN)

    def denormalize_output(self, arr):
        '''
        Denormalizes the output array using self.MN and self.MX (derived from the training data)
        Args:
            arr: numpy array to denormalize
        
        Returns:
            numpy array: denormalized array
        '''
        # NOTE: assumes outputs are the last 2 columns, should generalize
        return arr * (self.MX[-2:] - self.MN[-2:]) + self.MN[-2:]

    def load_trained_network(self, weights_csv, params_json, minmax_json='scaling_values.json'):
        '''
        Loads a trained network from weights and parameters files, and automatically
        loads min/max scaling values from a JSON file for denormalization later on.
        Args:
            weights_csv: path to CSV file containing weights and biases
            params_json: path to JSON file containing network parameters
            minmax_json: path to JSON file containing min/max scaling values
        
        Returns:
            Network: instance of the Network class with loaded weights and biases
        '''
        self.load_min_max(minmax_json)

        with open(params_json, "r") as f:
            params = json.load(f)

        hidden = int(params["hidden_neurons"])
        eta = float(params["eta"])
        alpha = float(params["alpha"])

        #NOTE: Assumes 2 inputs and 2 outputs, should generalize
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

            for row in reader:
                L = int(row["layer"])
                N = int(row["neuron"])
                idx = int(row["index"])
                val = float(row["value"])
                typ = row["param_type"]

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


def process_data(file_path="ce889_dataCollection.csv"):
    '''
    Full data processing pipeline:
    1. Load CSV
    2. Shuffle
    3. Split into train/val/test
    4. Compute min/max from training set
    5. Normalize all sets
    6. Save normalized sets and scaling values
    Args:
        file_path: path to the CSV data file, defaults to "ce889_dataCollection.csv"
    '''
    processor = DataProcessor(file_path)

    # Load file via static load_csv
    data = DataProcessor.load_csv(file_path)
    # Set the data from the file as the data attribute of the instance
    processor.data = data
    print(f"Data shape: {data.shape}")

    # Shuffle via static index shuffler
    shuffled_indices = DataProcessor.shuffle_indices(len(data))
    data = data[shuffled_indices]
    processor.data = data
    print("Data shuffled.")

    # Data split
    train, val, test = processor.split_data()
    print("Split complete:")
    print("  Train:", train.shape)
    print("  Val:  ", val.shape)
    print("  Test: ", test.shape)

    # Min-Max Scaling
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
    print("\nMN:", processor.MN)
    print("MX:", processor.MX)

    print("\n=== Preprocessing Complete ===")