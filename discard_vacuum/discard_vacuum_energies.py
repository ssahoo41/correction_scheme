import numpy as np
import h5py
import pickle
import glob
import os
import time 
import sys
from sklearn.preprocessing import StandardScaler

class DiscardVacuum:
    def __init__(self, mcsh_max_order=2, mcsh_step_size=0.5, mcsh_max_r=3.0):
        """
        Initialize the DiscardVacuum class with MCSH parameters.
        
        Args:
            mcsh_max_order (int): Maximum MCSH order
            mcsh_step_size (float): Step size for MCSH
            mcsh_max_r (float): Maximum radius for MCSH
        """
        self.mcsh_max_order = mcsh_max_order
        self.mcsh_step_size = mcsh_step_size
        self.mcsh_max_r = mcsh_max_r
        self.hsmp_filenames, self.num_features = self._get_feature_list_hsmp()
        
    def _log_result(self, log_filename, message):
        """Log a message to a file."""
        with open(log_filename, 'a') as f:
            f.write(message)
            
    def _filepath_contains_spin(self, filepath):
        """Check if filepath contains 'spin'."""
        return "spin" in filepath.lower()
    
    def _get_feature_list_hsmp(self):
        """
        Get the list of HSMP features and count.
        
        Returns:
            tuple: (list of HSMP filenames, number of features)
        """
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
    
    def read_hdf5_data(self, filepath):
        """
        Read data from HDF5 file.
        
        Args:
            filepath (str): Path to HDF5 file
            
        Returns:
            np.ndarray: Feature array
        """
        with h5py.File(filepath, 'r') as data:
            functional_grp = data["functional_database/PBE0"]
            Nx, Ny, Nz = functional_grp["metadata/FD_GRID"][:]
            feature_grp = functional_grp["feature"]
            
            grid_points = Nx * Ny * Nz
            feature_arr = np.zeros((grid_points, self.num_features + 2))
            
            feature_arr[:, 0] = feature_grp["dens"][:]
            feature_arr[:, 1] = feature_grp["sigma.csv"][:]
            
            for i, feature in enumerate(self.hsmp_filenames):
                feature_arr[:, i + 2] = feature_grp[feature][:]
                
        return feature_arr
    
    def discard_vacuum(self, feature_arr, threshold=1e-5):
        """
        Filter out vacuum regions based on density threshold.
        
        Args:
            feature_arr (np.ndarray): Input feature array
            threshold (float): Density threshold for filtering
            
        Returns:
            np.ndarray: Filtered feature array
        """
        return feature_arr[feature_arr[:, 0] > threshold]
    
    def process_system(self, system_path):
        """
        Process a single system to discard vacuum regions.
        
        Args:
            system_path (str): Path to the system HDF5 file
        """
        system_name = system_path.split("/")[-1].split("_HSMP")[0]
        print(f"Discarding vacuum for {system_name}...")
        
        start_time = time.time()
        feature_arr = self.read_hdf5_data(system_path)
        filtered_feat = self.discard_vacuum(feature_arr)
        end_time = time.time()
        
        print(f"Total time: {end_time - start_time}")
        print(f"Filtered shape: {filtered_feat.shape}")
        
        # Save filtered data back to HDF5 file
        with h5py.File(system_path, 'a') as data:
            functional_grp = data["functional_database/PBE0"]
            if "filtered_feature" in functional_grp:
                del functional_grp["filtered_feature"]
            functional_grp.create_dataset("filtered_feature", data=filtered_feat)
            print(f"Discarded vacuum of {system_name}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <system_type> <system_path>")
        sys.exit(1)
        
    system_type = sys.argv[1]
    system_path = sys.argv[2]
    
    processor = DiscardVacuum()
    processor.process_system(system_path)

if __name__ == "__main__":
    main()