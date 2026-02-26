import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .activationfunctions import Sigmoid, Linear
from .neuron import Neuron
from .layer import Layer
from .network import Network

#TODO
def read_from_file(file_path):
    return pd.read_csv(file_path)

#TODO
def write_to_file(df: pd.DataFrame, file_name):
    df.to_csv(file_name)

def plot_rms_errors(net, train_rms, val_rms):
    """
    Plots the RMS training and RMS validation errors on the same graph,
    and includes network parameters (eta, alpha, hidden neurons) in the title.
    """
    
    # Convert to simple 1D arrays (flatten handles cases returned as nested)
    train_rms = np.array(train_rms, dtype=float).flatten()
    val_rms = np.array(val_rms, dtype=float).flatten()
    
    epochs = np.arange(1, len(train_rms) + 1)

    # Extract parameters from the network object
    eta = net.eta
    alpha = net.alpha
    hidden = net.num_hidden_neurons
    n_inputs = net.num_inputs
    n_outputs = net.num_outputs

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_rms, label='Training RMS', linewidth=2)
    plt.plot(epochs, val_rms, label='Validation RMS', linewidth=2)

    plt.xlabel('Epochs')
    plt.ylabel('RMS Error')

    plt.title(
        f"Training & Validation RMS vs Epochs\n"
        f"(η={eta}, α={alpha}, Hidden Neurons={hidden}, Inputs={n_inputs}, Outputs={n_outputs})"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_all_results(all_results):
    """
    Plots all training/validation RMS curves at once, one plot per configuration.
    """
    print("\nGenerating all comparison plots...\n")

    for idx, result in enumerate(all_results):
        net = result["net"]
        train_err = result["train_errors"]
        val_err = result["val_errors"]

        print(f"Plot {idx+1}/{len(all_results)}")

        plot_rms_errors(net, train_err, val_err)

def export_weights_readable(net, filename="trained_weights.csv"):
    rows = []
    for layer_index, layer in enumerate(net.layers):
        for neuron_index, neuron in enumerate(layer.neurons):
            for w_index, w in enumerate(neuron.weights):
                rows.append({
                    "layer": layer_index,
                    "neuron": neuron_index,
                    "weight_index": w_index,
                    "weight_value": w
                })

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)

#TODO
def train(net:Network, train_data, epochs=1):# Epochs here decides the number of epochs done before doing a validation set pass
    '''
    Trains the passed Network using the training data provided from a CSV file
    and the number of epochs specified 
    and returns a list containing the RMS errors of all epoch  
    '''
    X_train = np.array(train_data.iloc[:,:2], dtype=float)
    y_train = np.array(train_data.iloc[:,2:], dtype=float)

    # creates a list of tuples, each tuple representing an example from the training set (input and output pairs)
    train_examples = list(zip(X_train, y_train))

    rms_errors = net.train(train_examples)
    # print(f"RMS error of training pass = {rms_errors:.6f}")
 
    # rms_errors = []
    # for i in range(epochs):
    #     rms_errors.append(net.train(train_examples))

    #     if i == 0 or (i + 1) % 10 == 0:
    #         print(f"RMS error after {i+1} epochs = {rms_errors[-1]:.6f}")

    return rms_errors

#TODO
def validate(net: Network, val_data):
    '''
    Performs a Validation pass on the validation data provided from a CSV file
    on the Network and returns the RMS error of the validation pass
    '''
    X_val = np.array(val_data.iloc[:,:2], dtype=float)
    y_val = np.array(val_data.iloc[:,2:], dtype=float)
    
    val_examples = list(zip(X_val, y_val))

    rms_error = net.validate(val_examples)
    # print(f"RMS error of validation pass = {rms_error:.6f}")

    return rms_error

#TODO
def learn(net: Network, epochs=100):
    train_data = read_from_file('../normalized_training_data_ce889_dataCollection.csv')
    val_data = read_from_file('../normalized_validation_data_ce889_dataCollection.csv')

    train_rms_errors = []
    val_rms_errors = []
    for i in range(epochs):
        train_rms_errors.append(train(net, train_data))
        val_rms_errors.append(validate(net, val_data))
        
        # if i == 0 or (i + 1) % 10 == 0:
        #     print("Epoch", i)
        #     print()
        
    return train_rms_errors, val_rms_errors

#TODO
def test():
    test_data = read_from_file('../normalized_test_data_ce889_dataCollection.csv')

def early_stopping(val_errors, patience=10, min_delta=1e-5):
    """
    Returns True if validation error has not improved for 'patience' epochs.
    """
    if len(val_errors) < patience:
        return False

    recent = val_errors[-patience:]
    return max(recent) - min(recent) < min_delta

def optimize(
    num_hidden_neurons: list,
    etas: list,
    alphas: list,
    epochs: list,
    patience=15,
    log_csv="hyperparameter_search_log.csv"
):

    log_rows = []
    best_val_error = float("inf")
    best_params = {}
    best_net = None
    all_results = []   

    total_runs = len(num_hidden_neurons) * len(etas) * len(alphas) * len(epochs)
    run_number = 0
    for epoch in epochs:
        for hn in num_hidden_neurons:
            for eta in etas:
                for alpha in alphas:
                    run_number += 1
                    print(f"\n--- Running {run_number}/{total_runs} ---")
                    print(f"η={eta}, α={alpha}, hidden={hn}, epochs={epoch}")

                    net = Network(
                        num_inputs=2,
                        num_outputs=2,
                        num_hidden_neurons=hn,
                        eta=eta,
                        alpha=alpha
                    )

                    train_errors, val_errors = [], []
                    for e in range(epoch):
                        t_err, v_err = learn(net, 1)
                        train_errors.extend(t_err)
                        val_errors.extend(v_err)

                        if early_stopping(val_errors, patience=patience):
                            print("Early stopping triggered.")
                            break

                    final_val = val_errors[-1]

                    # Save for CSV
                    log_rows.append({
                        "eta": eta,
                        "alpha": alpha,
                        "hidden_neurons": hn,
                        "epochs_ran": len(val_errors),
                        "final_train_rms": train_errors[-1],
                        "final_val_rms": final_val
                    })

                    all_results.append({
                        "net": net,
                        "train_errors": train_errors.copy(),
                        "val_errors": val_errors.copy()
                    })

                    if final_val < best_val_error:
                        best_val_error = final_val
                        best_params = {
                            "eta": eta,
                            "alpha": alpha,
                            "hidden_neurons": hn,
                            "epochs_ran": len(val_errors),
                            "validation_rms": final_val
                        }
                        best_net = net

    df = pd.DataFrame(log_rows)
    df.to_csv(log_csv, index=False)
    print(f"\nSearch results saved to {log_csv}")

    export_weights_readable(best_net, "trained_weights.csv")

    df_params = pd.DataFrame([best_params])
    write_to_file(df_params, 'best_parameters.csv')

    print(best_params)

    return best_net, all_results

if __name__ == '__main__':
    net = Network(num_inputs=2, num_outputs=2, num_hidden_neurons=2, eta=0.1, alpha=0.9)
    
    best_net, all_results = optimize(
        num_hidden_neurons=[2, 4],
        etas=[1, 0.1, 0.01, 0.001],
        alphas=[0.1, 0.5, 0.9],
        epochs=[100],
    )

    # plot_all_results(all_results)
