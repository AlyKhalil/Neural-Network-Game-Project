# --- util.py ---
import numpy as np
import csv
import json
import matplotlib.pyplot as plt

from .network import Network
from .DataProcessor import DataProcessor, process_data


def slice_X_y(data: np.ndarray, net: Network):
    n_in = net.num_inputs
    X = data[:, :n_in]
    y = data[:, n_in:]
    return X, y


def train_one_epoch(net: Network, X, y):
    examples = list(zip(X, y))
    return net.train(examples)


def validate(net: Network, X, y):
    examples = list(zip(X, y))
    return net.validate(examples)


def test(net: Network, X, y):
    examples = list(zip(X, y))
    return net.test(examples)


def learn(net: Network, train_arr: np.ndarray, val_arr: np.ndarray, epochs=100):
    train_errors = []
    val_errors = []

    best_val_rms = float("inf")
    epoch_of_best = -1
    best_weights_snapshot = None

    # Split once
    n_in = net.num_inputs
    X_train = train_arr[:, :n_in]
    y_train = train_arr[:, n_in:]

    X_val = val_arr[:, :n_in]
    y_val = val_arr[:, n_in:]

    for epoch in range(epochs):

        # Index-based shuffle
        perm = DataProcessor.shuffle_indices(len(X_train), seed=epoch) # seed=epoch produces a deterministic shuffle
        X_train = X_train[perm]
        y_train = y_train[perm]

        train_rms = train_one_epoch(net, X_train, y_train)
        val_rms = validate(net, X_val, y_val)

        train_errors.append(train_rms)
        val_errors.append(val_rms)

        if val_rms < best_val_rms:
            best_val_rms = val_rms
            epoch_of_best = epoch + 1
            best_weights_snapshot = net.get_weights_snapshot()

        if early_stopping(val_errors):
            print("Early stopping triggered.")
            break

    # Restore best weights
    if best_weights_snapshot is not None:
        net.set_weights_snapshot(best_weights_snapshot)

    return train_errors, val_errors, best_val_rms, epoch_of_best


def early_stopping(val_errors, patience=30, min_delta=1e-4):
    if len(val_errors) < patience:
        return False
    recent = val_errors[-patience:]
    return max(recent) - min(recent) < min_delta


def plot_rms_errors(net, train_rms, val_rms):
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

                    t_err, v_err, best_val_rms, epoch_of_best = learn(
                        net, train_arr, val_arr, epochs=E
                    )

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
        best_test_rms = test(best_net, X_test, y_test)
        best_params["test_rms"] = float(best_test_rms)

        export_weights_readable(best_net, "trained_weights.csv")
    else:
        best_params["test_rms"] = None

    # Save logs
    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eta", "alpha", "hidden", "epoch_of_best", "best_val_rms"])
        writer.writerows(log_rows)

    with open("best_parameters.json", "w") as f:
        json.dump(best_params, f, indent=4)

    print("\nBest model params:")
    print(best_params)

    return best_net, all_results


if __name__ == '__main__':
    process_data()

    best_net, all_results = find_best_parameters(
        hidden_list=[1],
        eta_list=[0.001],
        alpha_list=[0.9],
        epoch_list=[2],
    )

    plot_all_results(all_results)
