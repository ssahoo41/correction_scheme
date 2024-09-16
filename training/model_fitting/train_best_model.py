import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
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
    """Sorts count array based on the order of molecules in CCSD(T) formation energy dictionary."""
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

def model_fitting(lasso_model_filepath, final_count_arr, target, systems):
    # ensemble models
    models = []
    num_splits = 5
    for i in range(num_splits):
        X_train, X_test, y_train, y_test = train_test_split(final_count_arr, target, test_size=0.2)
        scaler = StandardScaler()
        # save the scaler fitted to training data
        X_train = scaler.fit_transform(X_train)
        pickle.dump(scaler, open(f"scaler_pbe_{i}.pkl", "wb"))
        X_test = scaler.transform(X_test)
        lasso = Lasso(fit_intercept=False, max_iter=10000, selection='random', alpha=1e-3)
        lasso.fit(X_train, y_train)
        coef_filename = f"{i}_fold_coef.npy"
        np.save(coef_filename, lasso.coef_)
        models.append(lasso)
        y_pred = lasso.predict(X_test)
        print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
        dump(models, os.path.join(lasso_model_filepath, 'lasso_ensemble_models.joblib'))

def single_model_fitting(lasso_model_filepath, final_count_arr, target, systems):
    #X_train, X_test, y_train, y_test = train_test_split(final_count_arr, target, test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(final_count_arr)
    pickle.dump(scaler, open(f"scaler_pbe_true.pkl", "wb"))
    lasso = Lasso(fit_intercept=False, max_iter=10000, selection='random', alpha=1e-5)
    lasso.fit(X_train, target)
    coef_filename = "fold_coef_pbe_true.npy"
    np.save(coef_filename, lasso.coef_)
    y_pred = lasso.predict(X_train)
    print(f"MAE: {mean_absolute_error(target, y_pred)}")
    dump(lasso, os.path.join(lasso_model_filepath, 'best_lasso_model_pbe.joblib'))

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
    print(f"Shape of count array is {final_count_arr.shape}")
    # call the function for LASSO regression
    single_model_fitting(lasso_model_filepath, final_count_arr, target, systems)
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
    #stdscale = sys.argv[3]

    lasso_model_filepath = os.path.join("pbe_ensemble_models", f"model_all_{overall_sig}_sys_{sys_sig}_lasso")
    os.makedirs(lasso_model_filepath, exist_ok=True)

    # Execute the main function with the specified arguments
    main(overall_sig, sys_sig, ccsdt_file, pbe_file, atomic_number_file, count_path)