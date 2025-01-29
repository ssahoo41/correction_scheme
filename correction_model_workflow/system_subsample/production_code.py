import numpy as np
import h5py
import pickle
import os
import sys
from NNSubsampling import subsampling

class SystemSubsampler:
    def __init__(self, system_path, cutoff_sig, mcsh_max_order=2, mcsh_step_size=0.5, mcsh_max_r=3.0, verbose=False):
        self.system_path = system_path
        self.cutoff_sig = cutoff_sig
        self.mcsh_max_order = mcsh_max_order
        self.mcsh_step_size = mcsh_step_size
        self.mcsh_max_r = mcsh_max_r
        self.verbose = verbose
        self.base_dir = "subsampled_folder_ex"
        self.system_name = os.path.basename(system_path).split('.')[0]

    def log(self, message):
        if self.verbose:
            print(message)

    def get_feature_list_hsmp(self):
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

    def read_hdf5_data(self, hsmp_filenames):
        with h5py.File(self.system_path, 'r') as data:
            Nx, Ny, Nz = data["functional_database/PBE0/metadata/FD_GRID"][:]
            feature_arr = np.zeros((Nx * Ny * Nz, len(hsmp_filenames) + 2))
            feature_grp = data["functional_database/PBE0/feature"]
            feature_arr[:, 0] = feature_grp["dens"][:]
            feature_arr[:, 1] = feature_grp["sigma.csv"][:]
            for i, feature in enumerate(hsmp_filenames):
                feature_arr[:, i + 2] = feature_grp[feature][:]
            exx = feature_grp["exx"][:].reshape(-1,1)
            ex_lda = -(3/(4*np.pi)) * (3 * np.pi * np.pi * feature_arr[:, 0])**(1/3)
            ex_lda = ex_lda.reshape(-1,1)
        return feature_arr, ex_lda, exx

    def subsample_and_save(self, feature_arr, ex_lda, exx):
        subsampled_feature_arr, subsampled_ex_lda, subsampled_exx, len_sub = subsampling(
            data=feature_arr, cutoff_sig=self.cutoff_sig, rate=0.1, method="pykdtree", verbose=2, standard_scale=False)
        
        save_dir = os.path.join(self.base_dir, f"cutoff_{self.cutoff_sig}", self.system_name)
        os.makedirs(save_dir, exist_ok=True)
        
        pickle.dump(subsampled_feature_arr, open(os.path.join(save_dir, "features.pkl"), "wb"))
        pickle.dump(subsampled_ex_lda, open(os.path.join(save_dir, "ex_lda.pkl"), "wb"))
        pickle.dump(subsampled_exx, open(os.path.join(save_dir, "exx.pkl"), "wb"))
        
        self.log(f"Subsampled {len_sub} features for {self.system_name}.")

    def run(self):
        hsmp_filenames, _ = self.get_feature_list_hsmp()
        feature_arr, ex_lda, exx = self.read_hdf5_data(hsmp_filenames)
        self.subsample_and_save(feature_arr, ex_lda, exx)
        self.log(f"Completed processing {self.system_name}.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script.py <system_path> <cutoff_sig> <verbose>")
        sys.exit(1)
    
    system_path = sys.argv[1]
    cutoff_sig = float(sys.argv[2])
    verbose = sys.argv[3].lower() in ['true', 'yes', '1']

    subsampler = SystemSubsampler(system_path, cutoff_sig, verbose=verbose)
    subsampler.run()
