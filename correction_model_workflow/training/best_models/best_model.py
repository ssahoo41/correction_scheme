import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

def single_model_fitting(lasso_model_filepath, final_count_arr, target, systems, best_alpha):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(final_count_arr)
    # save scaler in lasso_model_filepath
    lasso = Lasso(fit_intercept=False, max_iter=20000, selection='random', alpha=best_alpha)
    lasso.fit(X_train, target)
    pickle.dump(scaler, open(os.path.join(lasso_model_filepath, f"scaler.pkl"), "wb"))
    coef_filename = os.path.join(lasso_model_filepath, "fold_coef.npy")
    np.save(coef_filename, lasso.coef_)
    y_pred = lasso.predict(X_train)
    print(f"MAE: {mean_absolute_error(target, y_pred)}")
    dump(lasso, os.path.join(lasso_model_filepath, 'best_lasso_model.joblib'))

def main(overall_sig, cutoff_sig, ccsdt_file, pbe_file, atomic_number_file, count_path, best_alpha):
    # Load energy and atomic number data
    ccsdt_energy = load_json(ccsdt_file)
    pbe_energy = load_json(pbe_file)
    atomic_number_dict = load_json(atomic_number_file)

    target_dict = calculate_target_variable(ccsdt_energy, pbe_energy, atomic_number_dict)
    print(np.mean(abs(np.array(list(target_dict.values())))))
    count_file = os.path.join(count_path, f"count_array_overall_{overall_sig}_system_{cutoff_sig}.csv")
    final_count_arr, target, systems = sort_count_array(count_file, target_dict)
    print(f"Shape of count array is {final_count_arr.shape}")
    # call the function for LASSO regression
    single_model_fitting(lasso_model_filepath, final_count_arr, target, systems, best_alpha)
    return

# Script Execution Entry Point
if __name__ == "__main__":
    ccsdt_file = "ccsdt_energy.json"
    pbe_file = "pbe_energy.json"
    atomic_number_file = "atoms_count_mat.json"

    mcsh_order = int(sys.argv[1])
    rcut = float(sys.argv[2])
    overall_sig = float(sys.argv[3])
    sys_sig = float(sys.argv[4])
    best_alpha = float(sys.argv[5])

    base_path = "/storage/home/hcoda1/0/ssahoo41/cedar_storage/ssahoo41/exact_exchange_work/thesis_datagen/publication_purpose/subsampling_script/partitioning"
    count_path = os.path.join(base_path, f"mcsh_{mcsh_order}_rcut_{rcut}")
    print(f"Count path: {count_path}")

    lasso_model_filepath = os.path.join("train_best_model_minmax_form", f"mcsh_{mcsh_order}_rcut_{rcut}", f"model_all_{overall_sig}_sys_{sys_sig}_lasso")
    os.makedirs(lasso_model_filepath, exist_ok=True)
    # save enerything in lasso_model_filepath
    # Execute the main function with the specified arguments
    main(overall_sig, sys_sig, ccsdt_file, pbe_file, atomic_number_file, count_path, best_alpha)