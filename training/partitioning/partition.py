import numpy as np
import pickle
import h5py
from pykdtree.kdtree import KDTree
from sklearn.preprocessing import StandardScaler
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

# partitioning should be done with filtered features only- to reduce the max distance
def read_hdf5_data(filepath, num_features, hsmp_filenames, no_vac_discard):
    with h5py.File(filepath, 'r') as data:
        filtered_feat = data["functional_database/PBE/filtered_feature"][:]
    return filtered_feat

def feature_scaling(feature_arr):
    rcut = np.arange(0.5, 4.5, 0.5)
    mcsh_order = np.arange(0, 5, 1)
    index = 2
    for order in mcsh_order:
        for rc in rcut:
            feature_arr[:, index] = feature_arr[:, index] * (rc**3)
            index += 1
    return feature_arr

def partition(data, refdata):
    kd_tree = KDTree(refdata,leafsize=6)
    distances, indices = kd_tree.query(data, k=1)
    indices, counts = np.unique(indices, return_counts=True)
    count_arr = np.zeros(len(refdata))
    for i, index in enumerate(indices):
        count_arr[index] = counts[i]
    max_distance = np.max(distances)
    print(max_distance)
    
    return count_arr, max_distance

# main code starts here
overall_sig = sys.argv[1]
system_sig = sys.argv[2]
mol_filepath = "/storage/cedar/cedar0/cedarp-amedford6-0/ssahoo41/exact_exchange_work/test_2_dir/data_preparation/hdf5_v2_data/molecules"
overall_refdata_path = "/storage/cedar/cedar0/cedarp-amedford6-0/ssahoo41/exact_exchange_work/NNS_subsampling/overall_subsample/overall_subsample_mcsh_4_True"
mol_files = os.listdir(mol_filepath)
for file in mol_files:
    if filepath_contains_spin(file):
        mol_files.remove(file)

systems = [mol_file.split("_HSMP_")[0] for mol_file in mol_files]

mcsh_max_order = 4
mcsh_step_size = 0.5
mcsh_max_r = 4.0
hsmp_filenames, num_features = get_feature_list_hsmp(mcsh_max_order, mcsh_step_size, mcsh_max_r)

refdata_path = os.path.join(overall_refdata_path, f"subsampled_{overall_sig}_system_{system_sig}.pkl")
refdata = pickle.load(open(refdata_path, "rb" ))
# refdata to array
refdata = np.array(refdata)

scaler = StandardScaler()
refdata_scaled = scaler.fit_transform(refdata)
#refdata = np.vstack((refdata, np.zeros(len(refdata[0]))))

count_arr = np.zeros((len(systems), len(refdata)))
max_distance = 0

start_time = time.time()
for i, mol_file in enumerate(mol_files):
    print("start processing system {}".format(systems[i]))
    system_path = os.path.join(mol_filepath, mol_file)
    if not filepath_contains_spin(system_path):
        temp_feature_arr = read_hdf5_data(system_path, \
                                        num_features,\
                                        hsmp_filenames, False)
        temp_feature_arr = feature_scaling(temp_feature_arr)
        temp_feature_arr_scaled = scaler.transform(temp_feature_arr)
        count_arr[i], temp_max_distance = partition(temp_feature_arr_scaled, refdata_scaled)
        max_distance = max(max_distance, temp_max_distance)
end_time = time.time()
print(f"Time taken: {end_time - start_time}")
print(f"max distance over all systems: {max_distance}")
with open(f'count_array_overall_{overall_sig}_system_{system_sig}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i, system in enumerate(systems):
        writer.writerow([i, system] + count_arr[i].tolist())
