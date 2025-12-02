# import numpy as np
# import csv
# import json
# import random
# import matplotlib.pyplot as plt

# from .network import Network
# from .DataProcessor import DataProcessor, process_data # DataProcessor -> class; process_data -> function

# def load_csv(path):
#     """Reads a numeric CSV (no header) into a list of float lists using only the stdlib."""
#     data = []
#     with open(path, "r", newline="") as f:
#         reader = csv.reader(f)
#         for row in reader:
#             # if not row:
#             #     continue
#             floats = [float(x) for x in row]
#             data.append(floats)
    
#     return np.array(data, dtype=float)

# def save_csv(arr, path):
#     """Save numpy array → CSV."""
#     with open(path, "w", newline="") as f:
#         writer = csv.writer(f)
#         for row in arr:
#             writer.writerow(row)

# def silce_X_y(data: np.ndarray, net: Network):
#     """Split dataset by columns based on network dimensions."""
#     n_in = net.num_inputs
#     X = data[:, :n_in]
#     y = data[:, n_in:]
#     return X, y

# def shuffle(data_arr, seed=42):
#     random.seed(seed)
#     n = len(data_arr)
#     indices = list(range(n))

#     for i in range(n - 1, 0, -1):
#         j = random.randint(0, i)
#         indices[i], indices[j] = indices[j], indices[i]
    
#     data_arr = data_arr[indices]
#     return data_arr

# def train_one_epoch(net: Network, train_arr: np.ndarray):
#     """Run one epoch of training and return RMS error."""
#     X, y = silce_X_y(train_arr, net)
#     examples = list(zip(X, y))
#     return net.train(examples)

# def validate(net: Network, val_arr: np.ndarray):
#     """Run validation pass and return RMS error."""
#     X, y = silce_X_y(val_arr, net)
#     examples = list(zip(X, y))
#     return net.validate(examples)

# def test(net: Network, test_arr: np.ndarray):
#     """Run test pass and return RMS error."""
#     X, y = silce_X_y(test_arr, net)
#     examples = list(zip(X, y))
#     return net.test(examples)

# def learn(net: Network, train_arr: np.ndarray, val_arr: np.ndarray, epochs=100):
#     """Train network and return train/val RMS error lists."""

#     train_errors = []
#     val_errors = []

#     for _ in range(epochs):
#         train_arr = shuffle(train_arr) # Shuffle training examples each epoch
#         train_errors.append(train_one_epoch(net, train_arr))
#         val_errors.append(validate(net, val_arr))
        
#         if early_stopping(val_errors):
#             print("Early stopping triggered.")
#             break

#     return train_errors, val_errors

# def early_stopping(val_errors, patience=30, min_delta=1e-4):
#     """Stop if validation error hasn't improved meaningfully."""
#     if len(val_errors) < patience:
#         return False
#     recent = val_errors[-patience:]
#     return max(recent) - min(recent) < min_delta

# def plot_rms_errors(net, train_rms, val_rms):
#     train_rms = np.array(train_rms).flatten()
#     val_rms = np.array(val_rms).flatten()
#     epochs = np.arange(1, len(train_rms) + 1)

#     plt.figure(figsize=(10, 6))
#     plt.plot(epochs, train_rms, label='Train RMS', linewidth=2)
#     plt.plot(epochs, val_rms, label='Validation RMS', linewidth=2)

#     plt.xlabel('Epoch')
#     plt.ylabel('RMS Error')
#     plt.title(f"RMS Error (η={net.eta}, α={net.alpha}, Hidden={net.num_hidden_neurons})")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# # def plot_rms_errors(train_rms, val_rms, title="RMS Error"):
# #     plt.figure(figsize=(12, 6))
# #     plt.plot(train_rms, label="Train RMS")
# #     plt.plot(val_rms, label="Validation RMS")
# #     plt.xlabel("Epoch")
# #     plt.ylabel("RMS Error")
# #     plt.title(title)
# #     plt.legend()
# #     plt.grid(True)
# #     plt.show()

# def plot_all_results(all_results):
#     for i, result in enumerate(all_results):
#         print(f"Plot {i+1}/{len(all_results)}")
#         plot_rms_errors(result["net"], result["train"], result["val"])

# def export_weights_readable(net, filename="trained_weights.csv"):
#     """Export all weights and biases in human-readable CSV."""
#     with open(filename, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["layer", "neuron", "param_type", "index", "value"])

#         for L_i, layer in enumerate(net.layers):
#             for N_i, neuron in enumerate(layer.neurons):
#                 for w_i, w in enumerate(neuron.weights):
#                     writer.writerow([L_i, N_i, "weight", w_i, float(w)])
#                 writer.writerow([L_i, N_i, "bias", 0, float(neuron.bias)])

# def find_best_parameters(
#     hidden_list,
#     eta_list,
#     alpha_list,
#     epoch_list,
#     train_path="normalized_training_data_ce889_dataCollection.csv",
#     val_path="normalized_validation_data_ce889_dataCollection.csv",
#     test_path="normalized_test_data_ce889_dataCollection.csv",
#     log_csv="hyperparameter_search_log.csv"
# ):

#     all_results = []
#     log_rows = []

#     train_arr = load_csv(train_path)
#     val_arr   = load_csv(val_path)

#     test_arr = load_csv(test_path)
#     best_test = float("inf")
#     best_net = None
#     best_params = {}

#     total = len(hidden_list) * len(eta_list) * len(alpha_list) * len(epoch_list)
#     run = 0

#     for E in epoch_list:
#         for H in hidden_list:
#             for ETA in eta_list:
#                 for ALPHA in alpha_list:

#                     run += 1
#                     print(f"\n--- Run {run}/{total} ---")
#                     print(f"Hidden={H}, η={ETA}, α={ALPHA}, Epochs={E}")

#                     net = Network(
#                         num_inputs=2, 
#                         num_outputs=2,
#                         num_hidden_neurons=H, 
#                         eta=ETA, 
#                         alpha=ALPHA
#                         )

#                     train_errors = []
#                     val_errors = []

#                     # train for E epochs
#                     t_err, v_err = learn(net, train_arr, val_arr, epochs=E)
#                     # train_errors.append(t_err) # !!
#                     # val_errors.append(v_err) # !!

#                     test_rms = test(net, test_arr)

#                     # store result (robust)
#                     last_train_rms = t_err[-1] if (isinstance(t_err, (list, np.ndarray)) and len(t_err) > 0) else None
#                     epochs_ran = len(v_err) if isinstance(v_err, (list, np.ndarray)) else 0

#                     all_results.append({
#                         "net": net,
#                         "train": t_err,
#                         "val": v_err,
#                         "test_rms": test_rms
#                     })

#                     log_rows.append([ETA, ALPHA, H, epochs_ran, last_train_rms, test_rms])

#                     # update best
#                     if test_rms is not None and test_rms < best_test:
#                         best_test = test_rms
#                         best_net = net
#                         best_params = {
#                             "eta": ETA,
#                             "alpha": ALPHA,
#                             "hidden_neurons": H,
#                             "epochs_ran": epochs_ran,
#                             "test_rms": float(test_rms)
#                         }


#     # Save log CSV
#     with open(log_csv, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["eta", "alpha", "hidden", "epochs", "train_rms", "test_rms"])
#         writer.writerows(log_rows)

#     # Save best params JSON
#     with open("best_parameters.json", "w") as f:
#         json.dump(best_params, f, indent=4)

#     # Save best weights
#     export_weights_readable(best_net, "trained_weights.csv")

#     print("\nBest model params:")
#     print(best_params)

#     return best_net, all_results

# if __name__ == '__main__':
#     # process_data()

#     best_net, all_results = find_best_parameters(
#         hidden_list=[5],
#         eta_list=[0.05],
#         alpha_list=[0.7],
#         epoch_list=[10],
#     )

#     plot_all_results(all_results)


import numpy as np
import csv
import json
import random
import matplotlib.pyplot as plt

from .network import Network
from .DataProcessor import DataProcessor, process_data  # DataProcessor -> class; process_data -> function

def load_csv(path):
    data = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            floats = [float(x) for x in row]
            data.append(floats)
    return np.array(data, dtype=float)

def save_csv(arr, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in arr:
            writer.writerow(row)

def silce_X_y(data: np.ndarray, net: Network):
    n_in = net.num_inputs
    X = data[:, :n_in]
    y = data[:, n_in:]
    return X, y

def shuffle(data_arr, seed=42):
    random.seed(seed)
    n = len(data_arr)
    indices = list(range(n))
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        indices[i], indices[j] = indices[j], indices[i]
    return data_arr[indices]

def train_one_epoch(net: Network, train_arr: np.ndarray):
    X, y = silce_X_y(train_arr, net)
    examples = list(zip(X, y))
    return net.train(examples)

def validate(net: Network, val_arr: np.ndarray):
    X, y = silce_X_y(val_arr, net)
    examples = list(zip(X, y))
    return net.validate(examples)

def test(net: Network, test_arr: np.ndarray):
    X, y = silce_X_y(test_arr, net)
    examples = list(zip(X, y))
    return net.test(examples)

def learn(net: Network, train_arr: np.ndarray, val_arr: np.ndarray, epochs=100):
    """
    Train network for `epochs` and return two lists:
      - train_errors: list of train RMS per epoch
      - val_errors: list of val RMS per epoch
    """
    train_errors = []
    val_errors = []

    for _ in range(epochs):
        train_arr = shuffle(train_arr)  # Shuffle training examples each epoch
        train_errors.append(train_one_epoch(net, train_arr))
        val_errors.append(validate(net, val_arr))

        if early_stopping(val_errors):
            print("Early stopping triggered.")
            break

    return train_errors, val_errors

def early_stopping(val_errors, patience=30, min_delta=1e-4):
    if len(val_errors) < patience:
        return False
    recent = val_errors[-patience:]
    return max(recent) - min(recent) < min_delta

def plot_rms_errors(net, train_rms, val_rms):
    """
    net: Network (used for title)
    train_rms, val_rms: list[float] of equal length
    """
    # Defensive: convert to 1-D numpy arrays
    train_rms = np.array(train_rms, dtype=float).ravel()
    val_rms = np.array(val_rms, dtype=float).ravel()

    if train_rms.shape != val_rms.shape:
        # If shapes differ, truncate to shortest to avoid plotting errors
        min_len = min(len(train_rms), len(val_rms))
        train_rms = train_rms[:min_len]
        val_rms = val_rms[:min_len]

    epochs = np.arange(1, len(train_rms) + 1)

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

def export_weights_readable(net, filename="trained_weights.csv"):
    if net is None:
        raise ValueError("export_weights_readable: net is None")
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "neuron", "param_type", "index", "value"])
        for L_i, layer in enumerate(net.layers):
            for N_i, neuron in enumerate(layer.neurons):
                for w_i, w in enumerate(neuron.weights):
                    writer.writerow([L_i, N_i, "weight", w_i, float(w)])
                writer.writerow([L_i, N_i, "bias", 0, float(neuron.bias)])

def find_best_parameters(
    hidden_list,
    eta_list,
    alpha_list,
    epoch_list,
    train_path="normalized_training_data_ce889_dataCollection.csv",
    val_path="normalized_validation_data_ce889_dataCollection.csv",
    test_path="normalized_test_data_ce889_dataCollection.csv",
    log_csv="hyperparameter_search_log.csv"
):

    all_results = []
    log_rows = []

    train_arr = load_csv(train_path)
    val_arr   = load_csv(val_path)
    test_arr = load_csv(test_path)

    best_test = float("inf")
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

                    # train for E epochs -> returns lists of length epochs_ran
                    t_err, v_err = learn(net, train_arr, val_arr, epochs=E)

                    test_rms = test(net, test_arr)

                    # robust last_train and epochs_ran
                    last_train_rms = t_err[-1] if (isinstance(t_err, (list, np.ndarray)) and len(t_err) > 0) else None
                    epochs_ran = len(v_err) if isinstance(v_err, (list, np.ndarray)) else 0

                    all_results.append({
                        "net": net,
                        "train": t_err,
                        "val": v_err,
                        "test_rms": test_rms
                    })

                    log_rows.append([ETA, ALPHA, H, epochs_ran, last_train_rms, test_rms])

                    # update best
                    if test_rms is not None and test_rms < best_test:
                        best_test = test_rms
                        best_net = net
                        best_params = {
                            "eta": ETA,
                            "alpha": ALPHA,
                            "hidden_neurons": H,
                            "epochs_ran": epochs_ran,
                            "test_rms": float(test_rms)
                        }

    # Save log CSV
    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eta", "alpha", "hidden", "epochs", "train_rms", "test_rms"])
        writer.writerows(log_rows)

    # Save best params JSON
    with open("best_parameters.json", "w") as f:
        json.dump(best_params, f, indent=4)

    # Save best weights (guard if none)
    if best_net is not None:
        export_weights_readable(best_net, "trained_weights.csv")
    else:
        print("No best_net to export (no run improved best_test).")

    print("\nBest model params:")
    print(best_params)

    return best_net, all_results

if __name__ == '__main__':
    process_data()

    best_net, all_results = find_best_parameters(
        hidden_list=[8, 12, 16, 24, 32, 64, 128],
        eta_list=[0.01, 0.001, 0.002, 0.0001],
        alpha_list=[0.7, 0.9],
        epoch_list=[100],
    )

    plot_all_results(all_results)
