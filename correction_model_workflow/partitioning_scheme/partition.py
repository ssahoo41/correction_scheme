import numpy as np
import pickle
import h5py
from pykdtree.kdtree import KDTree
import csv
import sys 
import os
import time

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

def filepath_contains_spin(filepath):
    return "spin" in filepath.lower()

def read_hdf5_data(filepath, num_features, hsmp_filenames):
    with h5py.File(filepath, 'r') as data:
        # Accessing the groups and datasets
        Nx, Ny, Nz = data["functional_database/PBE0/metadata/FD_GRID"][:]
        # Pre-allocating arrays
        grid_points = Nx * Ny * Nz
        feature_arr = np.zeros((grid_points, num_features + 2))
        feature_grp = data["functional_database/PBE0/feature"]
        # Directly slicing data into pre-allocated arrays
        feature_arr[:, 0] = feature_grp["dens"][:]
        feature_arr[:, 1] = feature_grp["sigma.csv"][:]
        for i, feature in enumerate(hsmp_filenames):
            feature_arr[:, i + 2] = feature_grp[feature][:]
    return feature_arr

def feature_scaling(feature_arr):
    rcut = np.arange(0.5, 3.5, 0.5)
    mcsh_order = np.arange(0, 3, 1)
    index = 2
    for order in mcsh_order:
        for rc in rcut:
            feature_arr[:, index] = feature_arr[:, index] * (rc**3)
            index += 1
    return feature_arr

def partition(data, refdata):
    kd_tree = KDTree(refdata,leafsize=6)
    distances, indices = kd_tree.query(data, k=1)
    print(len(distances))
    indices, counts = np.unique(indices, return_counts=True)
    count_arr = np.zeros(len(refdata))
    for i, index in enumerate(indices):
        count_arr[index] = counts[i]
    max_distance = np.max(distances)
    print("max distance")
    print(max_distance)
    
    return count_arr, max_distance

# main code starts here
overall_sig = sys.argv[1]
system_sig = sys.argv[2]
mol_filepath = "/storage/cedar/cedar0/cedarp-amedford6-0/ssahoo41/exact_exchange_work/test_2_dir/data_preparation/hdf5_format_data/molecules"
overall_refdata_path = "/storage/cedar/cedar0/cedarp-amedford6-0/ssahoo41/exact_exchange_work/NNS_subsampling/overall_subsample/overall_subsample_True"
mol_files = os.listdir(mol_filepath)
for file in mol_files:
    if filepath_contains_spin(file):
        mol_files.remove(file)

systems = [mol_file.split("_HSMP_")[0] for mol_file in mol_files]

mcsh_max_order = 2
mcsh_step_size = 0.5
mcsh_max_r = 3.0
hsmp_filenames, num_features = get_feature_list_hsmp(mcsh_max_order, mcsh_step_size, mcsh_max_r)

refdata_path = os.path.join(overall_refdata_path, f"subsampled_{overall_sig}_system_{system_sig}.pkl")
refdata = pickle.load(open(refdata_path, "rb" ))
refdata = np.vstack((refdata, np.zeros(len(refdata[0]))))

count_arr = np.zeros((len(systems), len(refdata)))
max_distance = 0

start_time = time.time()
for i, mol_file in enumerate(mol_files):
    print("start processing system {}".format(systems[i]))
    system_path = os.path.join(mol_filepath, mol_file)
    if not filepath_contains_spin(system_path):
        temp_feature_arr = read_hdf5_data(system_path, \
                                        num_features,\
                                        hsmp_filenames)
        temp_feature_arr = feature_scaling(temp_feature_arr)
        count_arr[i], temp_max_distance = partition(temp_feature_arr, refdata)
        max_distance = max(max_distance, temp_max_distance)
end_time = time.time()
print(f"Time taken: {end_time - start_time}")
print(f"max distance over all systems: {max_distance}")
with open(f'count_array_overall_{overall_sig}_system_{system_sig}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i, system in enumerate(systems):
        writer.writerow([i, system] + count_arr[i].tolist())
