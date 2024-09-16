import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

Ha_to_eV = 27.21136

def load_json(file_path):
    """Loads data from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_formation_energy(energy_dict, atoms_count_array):
    """Calculates formation energy from energy dictionary and atoms count array."""

    assert len(energy_dict) == atoms_count_array.shape[0], "Mismatch in lengths of energy_dict and atoms_count_array"

    energy_array = np.array(list(energy_dict.values())) * Ha_to_eV
    molecules = list(energy_dict.keys())
    reg = LinearRegression(fit_intercept=False)
    reg.fit(atoms_count_array, energy_array)
    predicted_energy = reg.predict(atoms_count_array)
    formation_energy = energy_array - predicted_energy
    return {molecule: energy for molecule, energy in zip(molecules, formation_energy)}

def calculate_target_variable(ccsdt_energy, pbe_energy, atomic_number_dict):
    """Calculates target variable for model training."""
    ccsdt_formation_en = calculate_formation_energy(ccsdt_energy, np.array(list(atomic_number_dict.values())))
    pbe_formation_en = calculate_formation_energy(pbe_energy, np.array(list(atomic_number_dict.values())))

    target_dict = {key: ccsdt_formation_en[key] - pbe_formation_en.get(key, 0) for key in ccsdt_formation_en}

    return target_dict

def sort_count_array(count_file, target_dict):
    """Sorts count array based on the order of molecules in CCSDT formation energy dictionary."""
    df = pd.read_csv(count_file, header=None)
    target_dict = {key: target_dict[key] for key in target_dict if key in df[1].values} # Filter out molecules not in target_dict

    df['sort_order'] = df[1].map(lambda x: list(target_dict.keys()).index(x) if x in target_dict else None)
    df_sorted = df.sort_values(by='sort_order').iloc[:, 2:].drop(columns=['sort_order'])
    return df_sorted.to_numpy(), np.array(list(target_dict.values())), list(target_dict.keys())

def write_csv(filename, data, delimiter=',', header=None):
    """Writes data to a CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if header:
            writer.writerow(header)
        writer.writerows(data)

def perform_cross_validation(alpha, model_filepath, final_count_arr, target, systems):
    kf = KFold(n_splits=5, shuffle=False)
    metrics = {
        'train_mae': [], 'test_mae': [], 'all_mae': [],
        'train_mse': [], 'test_mse': [], 'all_mse': [],
        'non_zero_parameters': []
    }
    # we will append these lists with the maximum error in each fold
    max_y_test = []
    final_error_max_y_test = []
    molecule_max_error = []

    for i, (train_index, test_index) in enumerate(kf.split(systems)):
        X_train, X_test = final_count_arr[train_index], final_count_arr[test_index]
        # scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        y_train, y_test = target[train_index], target[test_index]
        # Fit the LASSO model
        reg = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000, selection='random')
        reg.fit(X_train, y_train)
        y_test_pred = reg.predict(X_test)
        max_test_idx = np.argmax(abs(y_test))
        max_y_test.append(y_test[max_test_idx])
        final_error_max_y_test.append(y_test[max_test_idx] - y_test_pred[max_test_idx])
        molecule_max_error.append(systems[test_index[max_test_idx]])

        final_count_arr_scaled = scaler.transform(final_count_arr)
        # Collect and store metrics
        collect_metrics(metrics, 'train', y_train, reg.predict(X_train))
        collect_metrics(metrics, 'test', y_test, reg.predict(X_test))
        collect_metrics(metrics, 'all', target, reg.predict(final_count_arr_scaled))
        
        # Save the model
        model_filename = os.path.join(model_filepath, f"alpha_{alpha}", f"{i}_fold_model.pickle")
        save_model(reg, model_filename)
        coef_filename = os.path.join(model_filepath, f"alpha_{alpha}", f"{i}_fold_coef.npy")
        np.save(coef_filename, reg.coef_)
    # appending more lists to the metrics dictionary
    metrics['max_y_test'] = max_y_test
    metrics['final_error_max_y_test'] = final_error_max_y_test
    metrics['molecule_max_error'] = molecule_max_error
    metrics['non_zero_parameters'].append(np.count_nonzero(reg.coef_))
    
    return metrics

def collect_metrics(metrics_dict, prefix, y_true, y_pred):
    metrics_dict[f'{prefix}_mae'].append(mean_absolute_error(y_true, y_pred))
    metrics_dict[f'{prefix}_mse'].append(mean_squared_error(y_true, y_pred))

def save_model(model, filename):
    with open(filename, "wb") as file:
        pickle.dump(model, file)

def log_max_error(alpha, metrics, log_filename):
    with open(log_filename, 'a') as log_file:
        log_file.write(f"{alpha}\t")
        for i in range(len(metrics['max_y_test'])):
            log_file.write(f"{i}\t{metrics['max_y_test'][i]}\t{metrics['final_error_max_y_test'][i]}\t{metrics['molecule_max_error'][i]}\t")
        log_file.write("\n")

def log_metrics(alpha, metrics, log_filename):
    with open(log_filename, 'a') as log_file:
        log_file.write(f"{alpha}\t{np.mean(metrics['train_mae'])}\t{np.std(metrics['train_mae'])}\t"
                       f"{np.min(metrics['train_mae'])}\t{np.max(metrics['train_mae'])}\t"
                       f"{np.mean(metrics['test_mae'])}\t{np.std(metrics['test_mae'])}\t"
                       f"{np.min(metrics['test_mae'])}\t{np.max(metrics['test_mae'])}\t"
                       f"{np.mean(metrics['all_mae'])}\t{np.std(metrics['all_mae'])}\t"
                       f"{np.min(metrics['all_mae'])}\t{np.max(metrics['all_mae'])}\t"
                       f"{metrics['test_mae']}\t{metrics['non_zero_parameters']}\n")

def log_results(filename, message):
    """Log results to a file."""
    with open(filename, 'a') as f:
        f.write(message)

def model_fitting(model_filepath, final_count_arr, target, systems):
    """ Fit a LASSO model to the data and return the model and the predictions."""
    # Generate alpha values for fine-tuning around 1e-3
    #alpha_list = [10**exp for exp in range(-4, -2, 1)]
    #alpha_list += [1e-3 + i*(1e-4) for i in range(-5, 6)]  # Adding more granularity around 1e-3

    #print(alpha_list)
    alpha_list = [10**exp for exp in range(-8,3)] 
    #alpha_list = [5e-6, 7.5e-6, 1e-5, 2.5e-5, 5e-5]
    for alpha in alpha_list:
        start_time = time.time()
        alpha_path = os.path.join(model_filepath, f"alpha_{alpha}")
        os.makedirs(alpha_path, exist_ok=True)
        print(f"==== Training model with alpha = {alpha} ====")
        metrics = perform_cross_validation(alpha, model_filepath, final_count_arr, target, systems)
        log_metrics(alpha, metrics, os.path.join(model_filepath, "overall_log.txt"))
        log_max_error(alpha, metrics, os.path.join(model_filepath, "max_error_log.txt"))
        end_time = time.time()
        print(f"Time taken for alpha = {alpha}: {end_time - start_time} seconds")
    return

def main(overall_sig, cutoff_sig, ccsdt_file, pbe_file, atomic_number_file, count_path):
    # Load energy and atomic number data
    ccsdt_energy = load_json(ccsdt_file)
    pbe_energy = load_json(pbe_file)
    atomic_number_dict = load_json(atomic_number_file)

    target_dict = calculate_target_variable(ccsdt_energy, pbe_energy, atomic_number_dict)
    #systems = list(target_dict.keys())
    print(np.mean(abs(np.array(list(target_dict.values())))))
    count_file = os.path.join(count_path, f"count_array_overall_{overall_sig}_system_{cutoff_sig}.csv")
    final_count_arr, target, systems = sort_count_array(count_file, target_dict)
    print(final_count_arr.shape)
    # call the function for LASSO regression
    model_fitting(lasso_model_filepath, final_count_arr, target, systems)
    return

# Script Execution Entry Point
if __name__ == "__main__":
    ccsdt_file = "ccsdt_energy.json"
    pbe_file = "pbe_energy.json"
    atomic_number_file = "atoms_count_mat.json"
    count_path = "/storage/home/hcoda1/0/ssahoo41/cedar_storage/ssahoo41/exact_exchange_work/NNS_subsampling/partitioning_scheme/pbe_csv_true"
    if len(sys.argv) < 3:
        print("Usage: python script.py <overall_sig> <cutoff_sig>")
        sys.exit(1)
    
    overall_sig = float(sys.argv[1])
    sys_sig = float(sys.argv[2])
    stdscale = sys.argv[3]

    lasso_model_filepath = os.path.join("detailed_models_lasso_pbe", f"stdscaler_{stdscale}", f"model_all_{overall_sig}_sys_{sys_sig}_lasso")
    #os.makedirs(lasso_model_filepath, exist_ok=True)

    # Execute the main function with the specified arguments
    main(overall_sig, sys_sig, ccsdt_file, pbe_file, atomic_number_file, count_path)