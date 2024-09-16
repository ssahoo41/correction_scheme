from NNSubsampling import subsampling
import numpy as np
import pickle
import glob
import os
import time
import sys

start = time.time() #to note how much time overall subsampling takes 

def log_result(log_filename, message):
    f = open(log_filename, 'a')
    f.write(message)
    f.close()
    return

overall_cutoff_sig = float(sys.argv[1])
system_cutoff_sig = float(sys.argv[2])
std_scale = sys.argv[3]

print(f"Overall cutoff sig: {overall_cutoff_sig}")
print(f"System cutoff sig: {system_cutoff_sig}")
print(f"Standard scale: {std_scale}")

base_dir = "/storage/cedar/cedar0/cedarp-amedford6-0/ssahoo41/exact_exchange_work/"
subsample_dir = "NNS_subsampling/system_subsample/subsampled_folder_v2_False"
sub_dir = f"molecules/std_scale_True/X_system_training_subsample/cutoff_{system_cutoff_sig}"
full_path = os.path.join(base_dir, subsample_dir, sub_dir)
data_list = []
for file in os.listdir(full_path): 
    filepath = os.path.join(full_path, file)
    temp = pickle.load(open(filepath, "rb"))
    temp = np.array(temp)
    print(temp.shape)
    data_list.append(temp)

overall_data = np.vstack(data_list)
print(f"Length of data before subsampling: {overall_data.shape}")

folder_path = f"./overall_subsample_mcsh_4_{std_scale}/" #standard scaling of overall subsample
os.makedirs(folder_path, exist_ok=True)

subsampled_data_filename = os.path.join(folder_path, f"subsampled_{overall_cutoff_sig}_system_{system_cutoff_sig}.pkl")
subsampled_feature_arr, indices = subsampling(overall_data, cutoff_sig=overall_cutoff_sig, rate=0.1, method = "pykdtree",\
                                    verbose = 2, standard_scale=True) #returns unscaled features 

pickle.dump(subsampled_feature_arr, open(subsampled_data_filename, "wb" ) )
end = time.time()
print("Time elapsed:", end-start)
length = len(subsampled_feature_arr)
print("end length: {}".format(length))

log_filename = os.path.join(folder_path, "overall_subsample_log.dat")
os.makedirs(folder_path, exist_ok=True)
message = "{}\t{}\t{}\n".format(overall_cutoff_sig, system_cutoff_sig, length)
log_result(log_filename, message)
print("Success overall subsampling with cutoff sig {}".format(overall_cutoff_sig))
