import h5py
import numpy as np
import csv
import os
import sys
from ase import Atoms
import re
from ase.data import atomic_numbers

class HDF5Writer:
    """
    Class to process and write output from DFT calculations run using SPARC into HDF5 format.
    """
    
    def __init__(self, filepath, system_type, system_name, hdf5_path):
        """
        Initialize the HDF5Writer class

        :param filepath = Path to the directory containing the system directories
        :param system_name = Name of the system being processed
        :param functional = Name of the functional being processed
        """
        self.filepath = filepath
        self.system_type = system_type
        self.system_name = system_name
        self.hdf5_path = hdf5_path
        self.bohr_to_angstrom = 0.52917721067  # Conversion factor from Bohr to Angstrom
        self.unit_cell_length = 20  # Unit cell length in Bohr for molecules

    def read_coords_ion(self):
        """
        Reads atomic coordinates from a .ion file and returns atomic numbers and positions.
        Assumes the file format as specified with ATOM_TYPE, N_TYPE_ATOM, and COORD_FRAC sections.
        """
        
        ion_filepath = os.path.join(self.filepath, self.system_name, "sprc-calc.ion")
        atomic_symbols = []
        positions = []
        with open(ion_filepath, "r") as file:
            lines = file.readlines()

        current_atom_type = None
        for i, line in enumerate(lines):
            line = line.strip()

            if line.startswith("ATOM_TYPE:"):
                current_atom_type = line.split(":")[1].strip()

            elif line.startswith("COORD_FRAC:"):
                n_atoms_index = lines[i-3]  # N_TYPE_ATOM line is 3 lines above COORD_FRAC
                n_atoms = int(n_atoms_index.split(":")[1].strip())

                for j in range(i + 1, i + 1 + n_atoms):
                    if lines[j].strip() == "":
                        break
                    x_frac, y_frac, z_frac = map(float, lines[j].split())
                    x = x_frac * self.unit_cell_length * self.bohr_to_angstrom
                    y = y_frac * self.unit_cell_length * self.bohr_to_angstrom
                    z = z_frac * self.unit_cell_length * self.bohr_to_angstrom
                    positions.append([x, y, z])
                    atomic_symbols.append(current_atom_type)   
        atoms = Atoms(symbols=atomic_symbols, positions=positions)
        return atoms.get_atomic_numbers(), atoms.get_positions()

    def get_feature_list_hsmp(self, max_mcsh_order, step_size, max_r):
        """
        Generates lists of filenames for spin paired HSMP files based on given MCSH parameters.
        Iterates over spherical harmonics orders and cutoff radii.

        :param max_mcsh_order: Maximum order of spherical harmonics.
        :param step_size: Step size for the radial cutoff.
        :param max_r: Maximum radial cutoff.
        :return: list of filenames. 
        """
        hsmp_filenames = []
        for l in range(max_mcsh_order + 1):
            rcut = step_size
            while rcut <= max_r:
                filename = f"HSMP_l_{l}_rcut_{rcut:.6f}_spin_typ_0.csv"
                hsmp_filenames.append(filename)
                rcut += step_size
        return hsmp_filenames

    def get_spin_feature_list_hsmp(self, max_mcsh_order, step_size, max_r):
        """
        Generates lists of filenames for spin up and spin down HSMP files based on given MCSH parameters.
        Iterates over spherical harmonics orders and cutoff radii.

        :param max_mcsh_order: Maximum order of spherical harmonics.
        :param step_size: Step size for the radial cutoff.
        :param max_r: Maximum radial cutoff.
        :return: Two lists of filenames for spin up and spin down.
        """
        hsmp_filenames_spin_up = []
        hsmp_filenames_spin_down = []

        for l in range(max_mcsh_order + 1):
            rcut = step_size
            while rcut <= max_r:
                # For spin up (spin_typ = 1)
                filename_up = f"HSMP_l_{l}_rcut_{rcut:.6f}_spin_typ_1.csv"
                hsmp_filenames_spin_up.append(filename_up)
                # For spin down (spin_typ = 2)
                filename_down = f"HSMP_l_{l}_rcut_{rcut:.6f}_spin_typ_2.csv"
                hsmp_filenames_spin_down.append(filename_down)
                rcut += step_size
    
        return hsmp_filenames_spin_up, hsmp_filenames_spin_down

    def read_csv(self, filepath):
        feature_list = []
        with open(filepath, "r") as fp:
            lines = fp.readlines()
        for line in lines:
            feature_list.append(float(line.split(",")[-1]))
        return feature_list

    def read_cube(self, Nx_, Ny_, Nz_, filepath):
        try:
            with open(filepath, "r") as fp:
                # Skip the first two lines
                lines = fp.readlines()[2:]
            
            # Read the number of atoms
            n_atom = int(lines[0].split()[0])
            Nx = int(lines[1].split()[0])
            Ny = int(lines[2].split()[0])
            Nz = int(lines[3].split()[0])
            #print(Nx, Ny, Nz, n_atom)
            if (Nx_ != Nx or Ny_ != Ny or Nz_ != Nz):
                print("ERROR: Vector in this file has different dimensions from the input.")
                return None

            # Skip lines for atom information
            density_lines = lines[4+n_atom:]
            density_list = []
            # Read data and fill the density array
            for i, num in enumerate(density_lines):
                values = num.split()
                for val in values:
                    density_list.append(float(val))
            #print(len(density_list))
        except FileNotFoundError:
            print(f"Cannot open file \"{filepath}\"")
            return None
        return density_list

    def list_into_array(self, Nx_, Ny_, Nz_, density_list):
        count = 0
        dens = np.zeros((Nx_ * Ny_ * Nz_))
        #print(Nx_, Ny_, Nz_)
        for i in range(0, Nx_):
            for j in range(0, Ny_):
                for k in range(0, Nz_):
                    dens[(i) + (j)*Nx_ + (k)*Nx_*Ny_] = density_list[count]
                    count+=1
        return dens

    def read_electron_density(self, Nx_, Ny_, Nz_):
        dens_filename = os.path.join(self.filepath, self.system_name, "sprc-calc.dens")
        dens_list = self.read_cube(Nx_, Ny_, Nz_, dens_filename)
        dens = self.list_into_array(Nx_, Ny_, Nz_, dens_list)
        modified_dens = np.array([1e-10 if value < 1e-10 else value for value in dens])
        return modified_dens

    def read_spin_electron_density(self, Nx_, Ny_, Nz_):
        densUp_filename = os.path.join(self.filepath, self.system_name, "sprc-calc.densUp")
        densUp_list = self.read_cube(Nx_, Ny_, Nz_, densUp_filename)
        densUp = self.list_into_array(Nx_, Ny_, Nz_, densUp_list)
        modified_densUp = np.array([1e-10 if value < 1e-10 else value for value in densUp])
        densDwn_filename = os.path.join(self.filepath, self.system_name, "sprc-calc.densDwn")
        densDwn_list = self.read_cube(Nx_, Ny_, Nz_, densDwn_filename)
        densDwn = self.list_into_array(Nx_, Ny_, Nz_, densDwn_list)
        modified_densDwn = np.array([1e-10 if value < 1e-10 else value for value in densDwn])
        return modified_densUp, modified_densDwn

    def read_sigma(self, path_string, group_name, functional_group, sigma_string):
        """
        Read sigma from sigma.csv file
        """
        if group_name not in functional_group:
            feature_grp = functional_group.create_group(group_name)
        else:
            feature_grp = functional_group[group_name]
        
        sigma_filename = os.path.join(self.filepath, self.system_name, path_string, sigma_string)
        sigma = self.read_csv(sigma_filename)
        feature_grp.create_dataset(sigma_string, data=sigma)
        return

    def store_hsmp_features(self, hsmp_filenames, path_string, group_name, functional_group):
        """
        Stores HSMP features from CSV files into HDF5 file under a specific group.
        :param hdf5_filename: Name of the HDF5 file.
        :param hsmp_filenames: List of HSMP filenames.
        :param group_name: "feat_spin, feat or feat_nlr"
        """
        if group_name not in functional_group:
            feature_grp = functional_group.create_group(group_name)
        else:
            feature_grp = functional_group[group_name]
        
        for feature in hsmp_filenames:
            feature_filename = os.path.join(self.filepath, self.system_name, path_string, feature)
            temp_data = self.read_csv(feature_filename)
            feature_grp.create_dataset(feature, data=temp_data)
        return

    def process_one_functional(self, hdf5_filename, spin):
        """
        Processes one functional and writes data to HDF5 format.
        Reads various parameters from output files and stores them in an HDF5 file.
        Handles the creation of datasets for electron density and exchange energy.
        """
        MCSH_RADIAL_TYPE = 1
        MCSH_MAX_R = 4.0
        MCSH_MAX_ORDER = 4
        MCSH_R_STEPSIZE = 0.5
        output_path = os.path.join(self.filepath, self.system_name)
        out_file = os.path.join(output_path, "sprc-calc.out")
        # Initialize variables
        #U = []
        #CELL = []
        FD_GRID = []
        output_energy = None

        # read the file
        try:
            with open(out_file, 'r') as file:
                lines = file.readlines()
    
            for index, line in enumerate(lines):
                if line.strip().startswith('FD_GRID'):
                    FD_GRID = [int(item) for item in line.split()[1:]]
                elif line.strip().startswith('Total free energy'):
                    output_energy = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
        except FileNotFoundError as e:
            print("Error reading output file: {e}")
            return 

        #U = np.array(U)
        #cell = np.array(CELL)

        if spin:
            densUp, densDwn = self.read_spin_electron_density(*FD_GRID)
            hsmp_filenames_spin_up, hsmp_filenames_spin_down = self.get_spin_feature_list_hsmp(MCSH_MAX_ORDER, MCSH_R_STEPSIZE, MCSH_MAX_R)
        else:
            dens = self.read_electron_density(*FD_GRID)
            hsmp_filenames = self.get_feature_list_hsmp(MCSH_MAX_ORDER, MCSH_R_STEPSIZE, MCSH_MAX_R)
    
        with h5py.File(hdf5_filename,'a') as data:
            functional_database_group = data.require_group("functional_database")
            functional_group = functional_database_group.require_group("PBE")

            #Metadata group
            metadata_grp = functional_group.create_group('metadata')
            metadata_grp.create_dataset("FD_GRID", data=FD_GRID)
            metadata_grp.create_dataset("Total_free_energy", data=output_energy)

            #Feature groups
            if spin:
                featureUp_grp  = functional_group.create_group('feature_spin_up')
                featureDwn_grp = functional_group.create_group('feature_spin_down')
                featureUp_grp.create_dataset("densUp",data=densUp)
                featureDwn_grp.create_dataset("densDwn",data=densDwn)

                self.read_sigma("feat", "feature_spin_up", functional_group, "sigma_up.csv")
                self.read_sigma("feat", "feature_spin_down", functional_group, "sigma_dn.csv")
                self.store_hsmp_features(hsmp_filenames_spin_up, "feat", "feature_spin_up", functional_group)
                self.store_hsmp_features(hsmp_filenames_spin_down, "feat", "feature_spin_down", functional_group)

            else:
                feature_grp = functional_group.create_group('feature')
                feature_grp.create_dataset("dens",data=dens)
                self.read_sigma("feat", 'feature', functional_group, "sigma.csv")
                self.store_hsmp_features(hsmp_filenames, "feat", 'feature', functional_group)
                
        return

    def check_spin(self):
        """
        Checks if the system is spin polarized.
        """
        spin = False
        return spin

    def process_system(self, MCSH_MAX_ORDER = 2, MCSH_MAX_R = 3.0):
        """
        Processes data for a given system and functional. Reads atomic information 
        and calls process_one_functional to handle functional-specific data.
        """
        print(f"\n==========\nstart system: {self.system_name}")
        spin = self.check_spin()
        atomic_numbers, coords = self.read_coords_ion()

        with h5py.File(self.hdf5_path,'w') as data:
            metadata_grp = data.create_group("metadata")
            metadata_grp.create_dataset("atomic_numbers", data=atomic_numbers)
            metadata_grp.create_dataset("atomic_coords", data=coords)
            metadata_grp.create_dataset("spin", data=spin)

        self.process_one_functional(hdf5_path, spin)
        print(f"finish system: {self.system_name}")
        return


if __name__ == "__main__":
    #filepath with raw dft data
    filepath = sys.argv[1]
    #system type: "single_atoms", "molecules", "bulks", "cubic_bulks"
    system_type = sys.argv[2]
    
    h5_path = f"./hdf5_data/{system_type}"
    h5_path.mkdir(parents=True, exist_ok=True)
    
    mcsh_max_order = 4
    mcsh_max_r = 4.0

    systems = os.listdir(filepath)
    
    for system in systems:
        hdf5_filename = f"{system}_HSMP_{mcsh_max_order}l_rcut_{mcsh_max_r:.6f}.h5"
        hdf5_path = os.path.join(h5_path, hdf5_filename)
        if not os.path.exists(hdf5_path):
            print("Processing system: {}".format(system))
            hdf5_writer = HDF5Writer(filepath, system_type, system, hdf5_path)
            hdf5_writer.process_system(mcsh_max_order, mcsh_max_r)