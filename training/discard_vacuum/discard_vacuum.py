# modify the hdf5 file to discard the vacuum region based on magnitude of density
import numpy as np
import h5py
import pickle
import glob
import os
import time 
import sys
from sklearn.preprocessing import StandardScaler
import numpy as np

def log_result(log_filename, message):
    f = open(log_filename, 'a')
    f.write(message)
    f.close()
    return

def filepath_contains_spin(filepath):
    return "spin" in filepath.lower()

# for spin paired molecules
def get_feature_list_hsmp(max_mcsh_order, step_size, max_r):
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

def read_hdf5_data(filepath, num_features, hsmp_filenames):
    with h5py.File(filepath, 'r') as data:
        # Accessing the groups and datasets
        functional_grp = data["functional_database/PBE"]
        Nx, Ny, Nz = functional_grp["metadata/FD_GRID"][:]
        feature_grp = functional_grp["feature"]
        # Pre-allocating arrays
        grid_points = Nx * Ny * Nz
        feature_arr = np.zeros((grid_points, num_features + 2))
        # Directly slicing data into pre-allocated arrays
        feature_arr[:, 0] = feature_grp["dens"][:]
        feature_arr[:, 1] = feature_grp["sigma.csv"][:]
        for i, feature in enumerate(hsmp_filenames):
            feature_arr[:, i + 2] = feature_grp[feature][:]
    return feature_arr

def discard_vacuum(feature_arr):
    dens = feature_arr[:, 0]
    filtered_feat = feature_arr[feature_arr[:, 0] > 1e-5]
    return filtered_feat

if len(sys.argv) < 1:
    print(
        "Usage: python discard_vacuum.py <system_path>")
    sys.exit(1)

#system_type = sys.argv[1]
system_path = sys.argv[1]
system_name = system_path.split("/")[-1].split("_HSMP")[0]

mcsh_max_order = 4
mcsh_step_size = 0.5
mcsh_max_r = 4.0
hsmp_filenames, num_features = get_feature_list_hsmp(mcsh_max_order, mcsh_step_size, mcsh_max_r)

print(f"Discarding vacuum for {system_name}...")
feature_arr = read_hdf5_data(system_path, num_features, hsmp_filenames)
start_time = time.time()
filtered_feat = discard_vacuum(feature_arr)
end_time = time.time()
print(f"total time: {end_time - start_time}")
print(filtered_feat.shape)
with h5py.File(system_path, 'a') as data:
    functional_grp = data["functional_database/PBE"]
    if "filtered_feature" in functional_grp:
        del functional_grp["filtered_feature"]
    functional_grp.create_dataset("filtered_feature", data=filtered_feat)
    print(f"Discarded vacuum of {system_name}...")