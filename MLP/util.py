import numpy as np
import csv
import json
import matplotlib.pyplot as plt

from .network import Network
from .DataProcessor import DataProcessor, process_data


def slice_X_y(data: np.ndarray, net: Network):
    '''
    Slices the given data into inputs X and targets y
    Args:
        data: np.array Object
        net: Network Object

    Returns:
        Tuple (X, y),
        X is the list of inputs,
        and y is the list of corresponding targets
    '''
    n_in = net.num_inputs
    X = data[:, :n_in]
    y = data[:, n_in:]
    return X, y


def train_one_epoch(net: Network, train_examples):
    '''
    Does One Epoch through the training examples.
    Args:
        net: Network Object
        train_examples: Iterable of (input, target) tuples

    Returns:
        RMS error over the epoch
    '''
    return net.train(train_examples)


def validate(net: Network, val_examples):
    '''
    Does One Validation pass through the validation examples.
    Args:
        net: Network Object
        val_examples: Iterable of (input, target) tuples
    
    Returns:
        RMS error over the validation set
    '''
    return net.validate(val_examples)


def test(net: Network, test_examples):
    '''
    Does testing through the test examples.
    Args:
        net: Network Object
        test_examples: Iterable of (input, target) tuples   
    
    Returns:
        RMS error over the test set
    '''
    return net.test(test_examples)


def learn(net: Network, train_arr: np.ndarray, val_arr: np.ndarray, epochs=100):
    '''
    Trains the network using the given training and validation arrays.
    Implements early stopping based on validation error.
    Saves the best weights based on best validation RMS error and restores them at the end.
    Args:
        net: Network Object
        train_arr: np.array Object for training data
        val_arr: np.array Object for validation data
        epochs: Maximum number of epochs to train
    
    Returns:
        Tuple (train_errors, val_errors, best_val_rms, epoch_of_best)
        train_errors: List of training RMS errors per epoch
        val_errors: List of validation RMS errors per epoch
        best_val_rms: Best validation RMS error achieved
        epoch_of_best: Epoch number (1-indexed) when best validation RMS was achieved
    '''
    train_errors = []
    val_errors = []

    best_val_rms = float("inf")
    epoch_of_best = -1
    best_weights_snapshot = None

    X_train, y_train = slice_X_y(train_arr, net)
    X_val, y_val = slice_X_y(val_arr, net)

    for epoch in range(epochs):

        shuffled_indices = DataProcessor.shuffle_indices(len(X_train), seed=None) # seed=epoch produces a deterministic shuffle for reproducibility, remove for true randomness
        
        train_examples = ((X_train[i], y_train[i]) for i in shuffled_indices) # Shuffling the training examples
        val_examples = ((X_val[i], y_val[i]) for i in range(len(X_val)))

        train_rms = train_one_epoch(net, train_examples)
        val_rms = validate(net, val_examples)

        train_errors.append(train_rms)
        val_errors.append(val_rms)

        if val_rms < best_val_rms:
            best_val_rms = val_rms
            epoch_of_best = epoch + 1 # 1-indexed
            best_weights_snapshot = net.get_weights_snapshot()

        if early_stopping(val_errors):
            print("Early stopping triggered.")
            break

    # Restore best weights
    if best_weights_snapshot is not None:
        net.set_weights_snapshot(best_weights_snapshot)

    return train_errors, val_errors, best_val_rms, epoch_of_best


def early_stopping(val_errors, patience=20, min_change=1e-4):
    '''
    Early stopping criterion based on validation errors.
    Args:
        val_errors: List of validation RMS errors per epoch
        patience: Number of epochs to consider for early stopping
        min_change: Minimum change in validation error to qualify as improvement
    
    Returns:
        True if early stopping criterion is met, False otherwise
    '''
    if len(val_errors) < patience:
        return False
    recent = val_errors[-patience:]
    return max(recent) - min(recent) < min_change


def plot_rms_errors(net, train_rms, val_rms):
    '''
    Plotting function for RMS errors.
    '''
    train_rms = np.array(train_rms).ravel()
    val_rms = np.array(val_rms).ravel()

    m = min(len(train_rms), len(val_rms))
    train_rms = train_rms[:m]
    val_rms = val_rms[:m]

    epochs = np.arange(1, m + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_rms, label='Train RMS', linewidth=2)
    plt.plot(epochs, val_rms, label='Validation RMS', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('RMS Error')
    plt.title(f"RMS Error (η={net.eta}, α={net.alpha}, Hidden={net.num_hidden_neurons})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_all_results(all_results):
    '''
    Plots RMS errors for all results in the list.
    '''
    for i, result in enumerate(all_results):
        print(f"Plot {i+1}/{len(all_results)}")

        # defensive sanity checks
        net = result.get("net")
        train = result.get("train")
        val = result.get("val")

        if net is None or train is None or val is None:
            print("Skipping result due to missing keys: ", result.keys())
            continue

        plot_rms_errors(net, train, val)


def export_weights(net, filename="trained_weights.csv"):
    '''
    Exports the network weights and biases to a CSV file.
    Args:
        net: Network Object
        filename: Output CSV filename
    '''
    if net is None:
        raise ValueError("export_weights: net is None")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "neuron", "param_type", "index", "value"])

        for L_i, layer in enumerate(net.layers):
            for N_i, neuron in enumerate(layer.neurons):
                for w_i, w in enumerate(neuron.weights):
                    writer.writerow([L_i, N_i, "weight", w_i, float(w)])
                writer.writerow([L_i, N_i, "bias", 0, float(neuron.bias)])


def grid_search(
    hidden_list,
    eta_list,
    alpha_list,
    epoch_list,
    train_path="normalized_training_data_ce889_dataCollection.csv",
    val_path="normalized_validation_data_ce889_dataCollection.csv",
    test_path="normalized_test_data_ce889_dataCollection.csv",
    log_csv="hyperparameter_search_log.csv"
):
    '''
    Performs grid search over hyperparameters. 
    Logs results to CSV, saves best model weights and parameters.
    Args:
        hidden_list: List of hidden neuron counts to try
        eta_list: List of learning rates to try
        alpha_list: List of momentum factors to try
        epoch_list: List of epoch counts to try
        train_path: Path to training data CSV
        val_path: Path to validation data CSV
        test_path: Path to test data CSV
        log_csv: Path to output log CSV file
    
    Returns:
        Tuple (best_net, all_results)
        best_net: Network Object with best validation performance
        all_results: List of all results dictionaries
    '''

    all_results = []
    log_rows = []

    # Load using DataProcessor's static method
    train_arr = DataProcessor.load_csv(train_path)
    val_arr   = DataProcessor.load_csv(val_path)
    test_arr  = DataProcessor.load_csv(test_path)

    best_val_global = float("inf")
    best_net = None
    best_params = {}

    total = len(hidden_list) * len(eta_list) * len(alpha_list) * len(epoch_list)
    run = 0

    for E in epoch_list:
        for H in hidden_list:
            for ETA in eta_list:
                for ALPHA in alpha_list:

                    run += 1
                    print(f"\n--- Run {run}/{total} ---")
                    print(f"Hidden={H}, η={ETA}, α={ALPHA}, Epochs={E}")

                    net = Network(
                        num_inputs=2,
                        num_outputs=2,
                        num_hidden_neurons=H,
                        eta=ETA,
                        alpha=ALPHA
                    )

                    t_err, v_err, best_val_rms, epoch_of_best = learn(net, train_arr, val_arr, epochs=E)

                    all_results.append({
                        "net": net,
                        "train": t_err,
                        "val": v_err,
                        "best_val_rms": best_val_rms,
                        "epoch_of_best": epoch_of_best
                    })

                    log_rows.append([ETA, ALPHA, H, epoch_of_best, best_val_rms])

                    if best_val_rms < best_val_global:
                        best_val_global = best_val_rms
                        best_net = net
                        best_params = {
                            "eta": ETA,
                            "alpha": ALPHA,
                            "hidden_neurons": H,
                            "epoch_of_best": epoch_of_best,
                            "best_val_rms": float(best_val_rms)
                        }

    # Test best model
    if best_net is not None:
        n_in = best_net.num_inputs
        X_test = test_arr[:, :n_in]
        y_test = test_arr[:, n_in:]
        test_examples = ((X_test[i], y_test[i]) for i in range(len(X_test)))

        best_test_rms = test(best_net, test_examples)
        best_params["test_rms"] = float(best_test_rms)

        export_weights(best_net, "trained_weights.csv")
    else:
        best_params["test_rms"] = None

    # Save logs
    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eta", "alpha", "hidden", "epoch_of_best", "best_val_rms"])
        writer.writerows(log_rows)

    # Save best parameters
    with open("best_parameters.json", "w") as f:
        json.dump(best_params, f, indent=4)

    print("\nBest model params:")
    print(best_params)

    return best_net, all_results
