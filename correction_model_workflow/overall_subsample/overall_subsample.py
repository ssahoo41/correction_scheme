from NNSubsampling import subsampling
import numpy as np
import pickle
import glob
import os
import time
from typing import List, Tuple

class OverallSubsampler:
    def __init__(self, 
                 overall_cutoff_sig: float,
                 system_cutoff_sig: float,
                 std_scale: bool,
                 base_dir: str = "/storage/cedar/cedar0/cedarp-amedford6-0/ssahoo41/exact_exchange_work/"):
        """
        Args:
            overall_cutoff_sig (float): Cutoff for the second level of subsampling
            system_cutoff_sig (float): Cutoff used in the first level of subsampling
            std_scale (bool): Whether to apply standard scaling
            base_dir (str): Base directory for data storage
        """
        self.overall_cutoff_sig = overall_cutoff_sig
        self.system_cutoff_sig = system_cutoff_sig
        self.std_scale = std_scale
        self.base_dir = base_dir
        self.start_time = None
        
        # dirs
        self.subsample_dir = "NNS_subsampling/system_subsample/subsampled_folder_vac_False"
        self.sub_dir = f"molecules/std_scale_True/X_system_training_subsample/cutoff_{self.system_cutoff_sig}"
        self.output_folder = f"./overall_subsample_{self.std_scale}/"
        
    def _log_result(self, log_filename: str, message: str) -> None:
        """Log results."""
        with open(log_filename, 'a') as f:
            f.write(message)
            
    def _load_data(self) -> np.ndarray:
        """Load and combine all subsampled data."""
        full_path = os.path.join(self.base_dir, self.subsample_dir, self.sub_dir)
        data_list = []
        
        for file in os.listdir(full_path):
            filepath = os.path.join(full_path, file)
            with open(filepath, "rb") as f:
                temp = pickle.load(f)
            temp = np.array(temp)
            print(f"Loaded array of shape: {temp.shape}")
            data_list.append(temp)
            
        overall_data = np.vstack(data_list)
        print(f"Combined data shape before subsampling: {overall_data.shape}")
        return overall_data
    
    def _save_results(self, subsampled_data: np.ndarray) -> None:
        """Save subsampled data and log results."""
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Save subsampled data
        filename = f"subsampled_{self.overall_cutoff_sig}_system_{self.system_cutoff_sig}.pkl"
        output_path = os.path.join(self.output_folder, filename)
        with open(output_path, "wb") as f:
            pickle.dump(subsampled_data, f)
            
        # Log results
        length = len(subsampled_data)
        log_filename = os.path.join(self.output_folder, "overall_subsample_log.dat")
        message = f"{self.overall_cutoff_sig}\t{self.system_cutoff_sig}\t{length}\n"
        self._log_result(log_filename, message)
        
    def process(self) -> Tuple[np.ndarray, float]:
        """
        Run the overall subsampling process.
        
        Returns:
            Tuple[np.ndarray, float]: Subsampled data array and elapsed time
        """
        self.start_time = time.time()
        
        # Load data
        overall_data = self._load_data()
        
        # Perform subsampling
        subsampled_data, indices = subsampling(
            data=overall_data,
            cutoff_sig=self.overall_cutoff_sig,
            rate=0.1,
            method="pykdtree",
            verbose=2,
            standard_scale=self.std_scale
        )
        
        # Save results
        self._save_results(subsampled_data)
        
        # Calculate and print statistics
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        print(f"Final subsampled length: {len(subsampled_data)}")
        print(f"Successfully completed overall subsampling with cutoff sig {self.overall_cutoff_sig}")
        
        return subsampled_data, elapsed_time
    

def main():
    """Main entry point for command line usage."""
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python script.py overall_cutoff_sig system_cutoff_sig std_scale")
        sys.exit(1)
        
    overall_cutoff_sig = float(sys.argv[1])
    system_cutoff_sig = float(sys.argv[2])
    std_scale = sys.argv[3].lower() == 'true'
    
    print(f"Overall cutoff sig: {overall_cutoff_sig}")
    print(f"System cutoff sig: {system_cutoff_sig}")
    print(f"Standard scale: {std_scale}")
    
    # Create and run subsampler
    subsampler = OverallSubsampler(
        overall_cutoff_sig=overall_cutoff_sig,
        system_cutoff_sig=system_cutoff_sig,
        std_scale=std_scale
    )
    subsampled_data, elapsed_time = subsampler.process()

if __name__ == "__main__":
    main()