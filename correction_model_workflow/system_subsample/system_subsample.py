from NNSubsampling import subsampling
import numpy as np
import h5py
import pickle
import glob
import os
import argparse
import sys
from typing import Tuple, List

class Subsampling:
    def __init__(self, system_type: str, system_path: str, cutoff_sig: float, 
                 no_vac_discard: bool = True, std_scale: bool = False):
        """
            system_type (str): Type of system
            system_path (str): Path to hdf5 file of system
            cutoff_sig (float): Cutoff for subsampling
            no_vac_discard (bool): If True, use original features; if False, use filtered features
            std_scale (bool): If True, standard scale the features
        """
        self.system_type = system_type
        self.system_path = system_path
        self.cutoff_sig = cutoff_sig
        self.no_vac_discard = no_vac_discard
        self.std_scale = std_scale
        self.system_name = self.system_path.split("/")[-1].split("_HSMP")[0]
        
        # MCSH parameters
        self.mcsh_max_order = 2
        self.mcsh_step_size = 0.5
        self.mcsh_max_r = 3.0
        
        # Get feature list
        self.hsmp_filenames, self.num_features = self._get_feature_list_hsmp()

    def _log_result(log_filename: str, message: str) -> None:
        """Log results to a file."""
        with open(log_filename, 'a') as f:
            f.write(message)

    def _filepath_contains_spin(filepath: str) -> bool:
        """Check if filepath contains 'spin'."""
        return "spin" in filepath.lower()

    def _get_feature_list_hsmp(self) -> Tuple[List[str], int]:
        """Get list of HSMP features and count."""
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

    def _read_hdf5_data(self) -> np.ndarray:
        """Read and process HDF5 data."""
        with h5py.File(self.system_path, 'r') as data:
            functional_grp = data["functional_database/PBE0"]
            Nx, Ny, Nz = functional_grp["metadata/FD_GRID"][:]
            grid_points = Nx * Ny * Nz
            feature_arr = np.zeros((grid_points, self.num_features + 2))
            feature_grp = functional_grp["feature"]
            feature_arr[:, 0] = feature_grp["dens"][:]
            feature_arr[:, 1] = feature_grp["sigma.csv"][:]
            for i, feature in enumerate(self.hsmp_filenames):
                feature_arr[:, i + 2] = feature_grp[feature][:]
            
            if self.no_vac_discard:
                return feature_arr
            else:
                return functional_grp["filtered_feature"][:]

    def _subsample_system(self, feature_arr: np.ndarray) -> Tuple[np.ndarray, int]:
        """Subsample the feature array."""
        subsampled_feature_arr, indices = subsampling(
            data=feature_arr, 
            cutoff_sig=self.cutoff_sig, 
            rate=0.1, 
            method="pykdtree", 
            verbose=2, 
            standard_scale=self.std_scale
        )
        len_sub = len(subsampled_feature_arr)
        print(f"Length of subsampled array: {len_sub}\n")
        return subsampled_feature_arr, len_sub

    def _scale_features(self, feature_arr: np.ndarray) -> np.ndarray:
        """Scale features based on radius cutoff."""
        rcut = np.arange(0.5, 3.5, 0.5)
        mcsh_order = np.arange(0, 3, 1)
        index = 2
        for order in mcsh_order:
            for rc in rcut:
                feature_arr[:, index] = feature_arr[:, index] * (rc**3)
                index += 1
        return feature_arr

    def process(self) -> None:
        """Main processing method."""
        print(f"Processing {self.system_name}...")
        
        # Read and process data
        if not self._filepath_contains_spin(self.system_path):
            feature_arr = self._read_hdf5_data()
        
        # Scale features
        feature_arr = self._scale_features(feature_arr)
        
        # Subsample
        feature_arr_subsample, len_arr = self._subsample_system(feature_arr)
        
        # Save results
        self._save_results(feature_arr_subsample, len_arr)
        
        print(f"Done processing {self.system_name}!")

    def _save_results(self, feature_arr_subsample: np.ndarray, len_arr: int) -> None:
        """Save processed results and logs."""
        # Create directory structure
        base_dir = f"subsampled_folder_vac_{self.no_vac_discard}/{self.system_type}/std_scale_{self.std_scale}/"
        X_dir = os.path.join(base_dir, f"X_system_training_subsample/cutoff_{self.cutoff_sig}")
        os.makedirs(X_dir, exist_ok=True)

        # Save subsampled data
        X_subsampled_filename = os.path.join(X_dir, f"{self.system_name}_subsample.pkl")
        with open(X_subsampled_filename, "wb") as f:
            pickle.dump(feature_arr_subsample, f)

        # Log results
        log_filename = f"{base_dir}log_subsample_cutoff_{self.cutoff_sig}.txt"
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        message = f"{self.system_name}\t{self.cutoff_sig}\t{len_arr}\n"
        self._log_result(log_filename, message)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process HSMP data with subsampling.")
    parser.add_argument("system_type", type=str, help="Type of system")
    parser.add_argument("system_path", type=str, help="Path to hdf5 file of system")
    parser.add_argument("cutoff_sig", type=float, help="Cutoff for subsampling")
    parser.add_argument("no_vac_discard", choices=["True", "False"], 
                       help="If True, use original features; if False, use filtered features")
    parser.add_argument("std_scale", choices=["True", "False"], 
                       help="If True, standard scale the features")
    
    args = parser.parse_args()
    args.no_vac_discard = args.no_vac_discard == "True"
    args.std_scale = args.std_scale == "True"
    return args

def main():
    """Main entry point."""
    args = parse_args()
    
    # Print configuration
    print(f"System type: {args.system_type}")
    print(f"System path: {args.system_path}")
    print(f"Cutoff sig: {args.cutoff_sig}")
    print(f"No vac discard: {args.no_vac_discard}")
    print(f"Std scale: {args.std_scale}")
    
    # Create and run subsampling
    processor = Subsampling(
        system_type=args.system_type,
        system_path=args.system_path,
        cutoff_sig=args.cutoff_sig,
        no_vac_discard=args.no_vac_discard,
        std_scale=args.std_scale
    )
    processor.process()

if __name__ == "__main__":
    main()