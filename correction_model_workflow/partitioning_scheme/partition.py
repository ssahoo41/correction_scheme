import numpy as np
from pykdtree.kdtree import KDTree
import pickle
import csv
import os
import time
import h5py

class Partitioner:
    def __init__(self, config):
        """        
        Args:
            config (dict): Configuration dictionary containing parameters
        """
        self.mcsh_max_order = config['mcsh_max_order']
        self.mcsh_step_size = config['mcsh_step_size']
        self.mcsh_max_r = config['mcsh_max_r']
        self.hsmp_filenames, self.num_features = self._get_feature_list_hsmp()
        
    def _get_feature_list_hsmp(self):
        """Get HSMP feature list for spin paired molecules."""
        hsmp_filenames = []
        num_features = 0
        for l in range(self.mcsh_max_order + 1):
            rcut = self.mcsh_step_size
            while rcut <= self.mcsh_max_r:
                filename = f"HSMP_l_{l}_rcut_{rcut:.6f}_spin_typ_0.csv"
                hsmp_filenames.append(filename)
                rcut += self.mcsh_step_size
                num_features += 1
        return hsmp_filenames, num_features
    
    def _read_hdf5_data(self, filepath):
        """Read feature data from HDF5 file."""
        with h5py.File(filepath, 'r') as data:
            Nx, Ny, Nz = data["functional_database/PBE0/metadata/FD_GRID"][:]
            grid_points = Nx * Ny * Nz
            feature_arr = np.zeros((grid_points, self.num_features + 2))
            feature_grp = data["functional_database/PBE0/feature"]
            
            feature_arr[:, 0] = feature_grp["dens"][:]
            feature_arr[:, 1] = feature_grp["sigma.csv"][:]
            
            for i, feature in enumerate(self.hsmp_filenames):
                feature_arr[:, i + 2] = feature_grp[feature][:]
        return feature_arr
    
    def _feature_scaling(self, feature_arr):
        """Scale features based on radial cutoffs."""
        rcut = np.arange(0.5, 3.5, 0.5)
        mcsh_order = np.arange(0, 3, 1)
        index = 2
        for order in mcsh_order:
            for rc in rcut:
                feature_arr[:, index] = feature_arr[:, index] * (rc**3)
                index += 1
        return feature_arr
    
    def _partition_data(self, data, refdata):
        """Partition data using KDTree."""
        kd_tree = KDTree(refdata, leafsize=6)
        distances, indices = kd_tree.query(data, k=1)
        
        indices, counts = np.unique(indices, return_counts=True)
        count_arr = np.zeros(len(refdata))
        for i, index in enumerate(indices):
            count_arr[index] = counts[i]
            
        max_distance = np.max(distances)
        print(f"Maximum distance: {max_distance}")
        
        return count_arr, max_distance
    
    def process_system(self, system_path):
        """Process a single system for partitioning."""
        feature_arr = self._read_hdf5_data(system_path)
        feature_arr = self._feature_scaling(feature_arr)
        return feature_arr

    def run_partitioning(self, h5_files, refdata_path, output_dir):
        """
        Run partitioning for all systems.
        
        Args:
            h5_files (list): List of HDF5 file paths
            refdata_path (str): Path to reference data pickle file
            output_dir (str): Directory to save results
        """
        # Load reference data
        with open(refdata_path, 'rb') as f:
            refdata = pickle.load(f)
        refdata = np.vstack((refdata, np.zeros(len(refdata[0]))))
        
        # Initialize arrays
        systems = [os.path.basename(f).split("_HSMP_")[0] for f in h5_files]
        count_arr = np.zeros((len(systems), len(refdata)))
        max_distance = 0
        
        # Process each system
        start_time = time.time()
        for i, h5_path in enumerate(h5_files):
            print(f"Processing system: {systems[i]}")
            
            feature_arr = self.process_system(h5_path)
            count_arr[i], temp_max_distance = self._partition_data(feature_arr, refdata)
            max_distance = max(max_distance, temp_max_distance)
            
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"Maximum distance over all systems: {max_distance}")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'partition_results.csv')
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i, system in enumerate(systems):
                writer.writerow([i, system] + count_arr[i].tolist())