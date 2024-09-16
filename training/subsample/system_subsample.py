from NNSubsampling import subsampling
import numpy as np
import h5py
import pickle
import glob
import os
import argparse
import sys
from typing import Tuple, List

def parse_args():
    parser = argparse.ArgumentParser(description="Process HSMP data with subsampling.")
    # positional arguments
    parser.add_argument("system_type", type=str, help="Type of system")
    parser.add_argument("system_path", type=str, help="Path to hdf5 file of system")
    parser.add_argument("cutoff_sig", type=float, help="Cutoff for subsampling")

    # using choices to restrict the values
    parser.add_argument("no_vac_discard", choices=["True", "False"], help="If True, use original features; if False, use filtered features")
    #parser.add_argument("std_scale", choices=["True", "False"], help="If True, standard scale the features; if False, do not standard scale the features")

    args = parser.parse_args()
    args.no_vac_discard = args.no_vac_discard == "True"
    #args.std_scale = args.std_scale == "True"
    return args

def log_result(log_filename: str, message: str) -> None:
    with open(log_filename, 'a') as f:
        f.write(message)

def filepath_contains_spin(filepath: str) -> bool:
    return "spin" in filepath.lower()

# for spin paired molecules
def get_feature_list_hsmp(max_mcsh_order: int, step_size: float, max_r: float) -> Tuple[List[str], int]:
    hsmp_filenames = []
    num_features = 0
    for l in range(max_mcsh_order + 1):
        rcut = step_size
        while rcut <= max_r:
            filename = f"HSMP_l_{l}_rcut_{rcut:.6f}_spin_typ_0.csv"
            hsmp_filenames.append(filename)
            rcut += step_size
            num_features += 1
    return hsmp_filenames, num_features

def read_hdf5_data(filepath: str, num_features: int, 
                    hsmp_filenames: List[str],\
                    no_vac_discard: bool=True) -> np.ndarray:
    with h5py.File(filepath, 'r') as data:
        functional_grp = data["functional_database/PBE"]
        Nx, Ny, Nz = functional_grp["metadata/FD_GRID"][:]
        grid_points = Nx * Ny * Nz
        feature_arr = np.zeros((grid_points, num_features + 2))
        feature_grp = functional_grp["feature"]
        feature_arr[:, 0] = feature_grp["dens"][:]
        feature_arr[:, 1] = feature_grp["sigma.csv"][:]
        for i, feature in enumerate(hsmp_filenames):
            feature_arr[:, i + 2] = feature_grp[feature][:]
        if no_vac_discard:
            return feature_arr
        else:
            return functional_grp["filtered_feature"][:]

def subsample_system(feature_arr: np.ndarray, cutoff_sig: float) -> Tuple[np.ndarray, int]:
    subsampled_feature_arr, indices = subsampling(data=feature_arr, cutoff_sig=cutoff_sig, rate=0.1, method = "pykdtree", verbose = 2, standard_scale=True)
    len_sub = len(subsampled_feature_arr) 
    print(f"length of subsampled array: {len_sub}\n")
    return subsampled_feature_arr, len_sub

def main():

    args = parse_args()
    print(f"System type: {args.system_type}")
    print(f"System path: {args.system_path}")
    print(f"Cutoff sig: {args.cutoff_sig}")
    print(f"No vac discard: {args.no_vac_discard}") # Always False (which means vacuum is discarded)

    system_name = args.system_path.split("/")[-1].split("_HSMP")[0]
    mcsh_max_order, mcsh_step_size, mcsh_max_r = 4, 0.5, 4.0
    hsmp_filenames, num_features = get_feature_list_hsmp(mcsh_max_order, mcsh_step_size, mcsh_max_r)

    print(f"Processing {system_name}...")
    if  not filepath_contains_spin(args.system_path):
        feature_arr = read_hdf5_data(args.system_path, num_features, hsmp_filenames, args.no_vac_discard)

    rcut = np.arange(0.5, 4.5, 0.5)
    mcsh_order = np.arange(0, 4, 1)
    index = 2
    for order in mcsh_order:
        for rc in rcut:
            feature_arr[:, index] = feature_arr[:, index] * (rc**3)
            index += 1
    feature_arr_subsample, len_arr = subsample_system(feature_arr, args.cutoff_sig)

    base_dir = f"subsampled_folder_v2_{args.no_vac_discard}/molecules/std_scale_True/"
    X_dir = os.path.join(base_dir, f"X_system_training_subsample/cutoff_{args.cutoff_sig}")

    # Ensure these directories exist
    os.makedirs(X_dir, exist_ok=True)

    # Construct file paths
    X_subsampled_filename = os.path.join(X_dir, f"{system_name}_subsample.pkl")
    pickle.dump(feature_arr_subsample, open(X_subsampled_filename, "wb" ) )
    print(f"Done processing {args.system_path}!")

    # Assuming system_type and cutoff_sig are defined earlier in your code
    log_directory = f"subsampled_folder_v2_{args.no_vac_discard}/molecules/std_scale_True/"
    log_filename = f"{log_directory}log_subsample_cutoff_{args.cutoff_sig}.txt"

    # Ensure the directory exists
    os.makedirs(log_directory, exist_ok=True)
    message = f"{system_name}\t{args.cutoff_sig}\t{len_arr}\n"
    log_result(log_filename, message)
    print(f"Done processing {system_name}!")

if __name__ == "__main__":
    main()