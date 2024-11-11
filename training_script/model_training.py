import csv
from datetime import datetime
import os
import sys
import json
import time
import pickle
import platform
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


class EnergyCorrectionFitter:
    def __init__(
        self,
        mcsh,
        rcut,
        overall_sig,
        cutoff_sig,
        ccsdt_file,
        pbe_file,
        atomic_number_file,
        count_path,
        stdscale=None,
    ):
        """Initialize EnergyCalculator with instance variables since we need to keep referencing these for logging"""
        # Constants
        self.Ha_to_eV = 27.21136 # Hartree to eV

        # Input files and paths
        self._ccsdt_file = ccsdt_file
        self._pbe_file = pbe_file
        self._atomic_number_file = atomic_number_file
        self._count_path = count_path

        # parameters for logging
        self._mcsh = mcsh
        self._rcut = rcut
        self._overall_sig = overall_sig
        self._cutoff_sig = cutoff_sig
        self._stdscale = stdscale

        # Data
        self._ccsdt_energy = None
        self._pbe_energy = None
        self._atomic_number_dict = None
        self._target_dict = None
        self._final_count_arr = None
        self._target = None
        self._systems = None

        # Model path with platform, date, and system information to track experiment results
        # Get Python version, OS, and datetime
        python_version = f"Python {sys.version.split(' ')[0]}"
        os_name = platform.system()
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        time_prefix = now.strftime("%H-%M")

        # Construct the directory prefix
        model_filepath_prefix = (
            f"{python_version} | {os_name} | {today} | {time_prefix}"
        )

        self.model_filepath = os.path.join(
            model_filepath_prefix,
            f"mcsh_{mcsh}_rcut_{rcut}",
            f"stdscaler_{stdscale}",
            f"model_all_{overall_sig}_sys_{sys_sig}_lasso",
        )
        os.makedirs(self.model_filepath, exist_ok=True)

    # Ha_to_eV = 27.21136

    def load_json(self, file_path):
        """Loads data from a JSON file and raises errors if file not found or decode error arise"""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            raise
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {file_path}")
            raise

    def calculate_formation_energy(self, energy_dict, atoms_count_array, model_filepath = None):
        """Calculates formation energy from energy dictionary and atoms count array.
        Args:
        energy_dict (dict): Dictionary of energies of molecules specified in atoms_count_array
        atoms_count_array (np.ndarray): Array of atom counts
        example: {"NH2NO": [2, 0, 2, 1], "C2H6N2O2_E-Azodioxymethane": [6, 2, 2, 2], "NH3O": [3, 0, 1, 1], "H2O2": [2, 0, 0, 2], "C2H6": [6, 2, 0, 0], ....}
        """

        assert (
            len(energy_dict) == atoms_count_array.shape[0]
        ), "Mismatch in lengths of energy_dict and atoms_count_array"

        energy_array = np.array(list(energy_dict.values())) * self.Ha_to_eV # convert to eV

        molecules = list(energy_dict.keys()) # list of molecules

        reg = LinearRegression(fit_intercept=False)
        reg.fit(atoms_count_array, energy_array)
        predicted_energy = reg.predict(atoms_count_array)
        formation_energy = energy_array - predicted_energy
        # log the different energies

        pd.DataFrame(energy_array).to_csv(os.path.join(self.model_filepath,"energy_array.csv")) # ccsdt or pbe
        atom_types = [f"Atom_{i}" for i in range(atoms_count_array.shape[1])]
        pd.DataFrame(atoms_count_array, index=molecules, columns=atom_types).to_csv(
            os.path.join(self.model_filepath,"atoms_count_array.csv")
        ) # atom counts per molecule
        
        pd.DataFrame(np.array(predicted_energy)).to_csv(os.path.join(self.model_filepath,"predicted_energy.csv")) # energy predicted from regression 

        pd.DataFrame(formation_energy).to_csv(os.path.join(self.model_filepath,"formation_energy.csv"))
        # Save molecules to formation energy mapping
        formation_energy_dict = pd.DataFrame(
            {"Molecule": molecules, "Formation Energy (eV)": formation_energy}
        ).set_index("Molecule")
        formation_energy_dict.to_csv("molecules_to_formation_energy.csv")
        
        return {
            molecule: energy for molecule, energy in zip(molecules, formation_energy)
        }

    def calculate_target_variable(self, ccsdt_energy, pbe_energy, atomic_number_dict):
        """Calculates target variable for model training."""
        ccsdt_formation_en = self.calculate_formation_energy(
            ccsdt_energy, np.array(list(atomic_number_dict.values()))
        )
        pbe_formation_en = self.calculate_formation_energy(
            pbe_energy, np.array(list(atomic_number_dict.values()))
        )

        # target is DIFFERENCE in formation energies between ccsdt and pbe
        target_dict = {
            key: ccsdt_formation_en[key] - pbe_formation_en.get(key, 0)
            for key in ccsdt_formation_en
        }

        # Convert to DataFrame properly
        df = pd.DataFrame.from_dict(
            target_dict, orient="index", columns=["Energy_Difference"]
        )

        # Save to csv as artefact
        p = os.path.join(self.model_filepath, "target_var_calculations.csv")
        df.to_csv(p)

        return target_dict

    def sort_count_array(self, count_file, target_dict):
        """Sorts count array based on the order of molecules in CCS-DT formation energy dictionary."""
        df = pd.read_csv(count_file, header=None)

        target_dict = {
            key: target_dict[key] for key in target_dict if key in df[1].values
        }  # Filter out molecules not in target_dict

        df["sort_order"] = df[1].map(
            lambda x: list(target_dict.keys()).index(x) if x in target_dict else None
        )

        # Add timestamp with format YYYY-MM-DD_HH-MM
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        # Save with timestamp in filename
        output_filename = os.path.join(self.model_filepath,
                                       f"sorted_count_array_{timestamp}.csv")
        # df_sorted.to_csv(output_filename, index=False)
        df_sorted = (
            df.sort_values(by="sort_order").iloc[:, 2:].drop(columns=["sort_order"])
        )
        df_sorted.to_csv(output_filename)  # artefact checking

        return (
            df_sorted.to_numpy(), #count arrays per molecule
            np.array(list(target_dict.values())), # target values
            list(target_dict.keys()), #molecules
        )

    def write_csv(self, filename, data, delimiter=",", header=None):
        """Writes data to a CSV file."""
        # csv_filepath = os.path.join(self.model_filepath, filename)

        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(
                csvfile, delimiter=delimiter, quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            if header:
                writer.writerow(header)
            writer.writerows(data)

        # with open(filename, "w", newline="") as csvfile:
        #     writer = csv.writer(
        #         csvfile, delimiter=delimiter, quotechar="|", quoting=csv.QUOTE_MINIMAL
        #     )
        #     if header:
        #         writer.writerow(header)
        #     writer.writerows(data)

    def perform_cross_validation(
        self, alpha, model_filepath, final_count_arr, target, systems
    ):
        kf = KFold(n_splits=5, shuffle=False)
        metrics = {
            "train_mae": [],
            "test_mae": [],
            "all_mae": [],
            "train_mse": [],
            "test_mse": [],
            "all_mse": [],
            "non_zero_parameters": [],
        }
        # we will append these lists with the maximum error in each fold
        max_y_test = []
        final_error_max_y_test = []
        molecule_max_error = []

        for i, (train_index, test_index) in enumerate(kf.split(systems)):
            X_train, X_test = final_count_arr[train_index], final_count_arr[test_index]

            # scale the data within the fold
            scaler = StandardScaler()
            X_train = scaler.fit_transform(
                X_train
            )  # LD: this learns the mean from the X_train
            self.write_csv(
                os.path.join(self.model_filepath, f"{alpha}_XTrain"), X_train)
            X_test = scaler.transform(
                X_test
            )  # applies mean from the X_train, not the test.
            y_train, y_test = target[train_index], target[test_index]

            # Reshapes to 2D array while preserving 1d aspect of target variable
            scaler = StandardScaler()
            y_train = scaler.fit_transform(y_train.reshape(-1, 1))
            y_test = scaler.transform(y_test.reshape(-1, 1))

            # Fit the LASSO model
            reg = Lasso(
                alpha=alpha, fit_intercept=False, max_iter=10000, selection="random"
            )
            reg.fit(X_train, y_train) # fitting step

            # store the number of non-zero parameters
            metrics["non_zero_parameters"].append(np.sum(reg.coef_ != 0))

            y_test_pred = reg.predict(X_test)
            # find the index of the maximum y_test value in this fold
            max_test_idx = np.argmax(y_test)
            max_y_test.append(y_test[max_test_idx])
            final_error_max_y_test.append(
                y_test[max_test_idx] - y_test_pred[max_test_idx]
            )
            molecule_max_error.append(systems[test_index[max_test_idx]])

            final_count_arr_scaled = scaler.transform(final_count_arr)

            # Get predictions
            y_train_pred = reg.predict(X_train)
            # y_test_pred = reg.predict(X_test)
            y_all_pred = reg.predict(final_count_arr_scaled)
            
            # Unscale predictions and actual values before metrics
            y_train_pred_unscaled = scaler.inverse_transform(y_train_pred.reshape(-1, 1)).ravel()
            y_test_pred_unscaled = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()
            
            # y_train and y_test are already 2D from earlier reshape
            y_train_unscaled = scaler.inverse_transform(y_train).ravel()
            y_test_unscaled = scaler.inverse_transform(y_test).ravel()
            y_train_unscaled = scaler.inverse_transform(y_train)
            y_train_pred_unscaled = scaler.inverse_transform(y_train_pred)
            
            y_test_unscaled = scaler.inverse_transform(y_test)
            y_test_pred_unscaled = scaler.inverse_transform(y_test_pred)
            
            # For "all" metrics, target is already unscaled so only unscale predictions
            y_all_pred_unscaled = scaler.inverse_transform(y_all_pred)
            
            # Collect and store metrics with unscaled values
            self.collect_metrics(metrics, "train", y_train_unscaled, y_train_pred_unscaled)
            self.collect_metrics(metrics, "test", y_test_unscaled, y_test_pred_unscaled)
            self.collect_metrics(metrics, "all", target, y_all_pred_unscaled)

            # # Collect and store metrics
            # self.collect_metrics(metrics, "train", y_train, reg.predict(X_train))
            # self.collect_metrics(metrics, "test", y_test, reg.predict(X_test))
            # self.collect_metrics(
            #     metrics, "all", target, reg.predict(final_count_arr_scaled)
            # )

            # Save the model
            model_filepath = os.path.join(
                model_filepath, f"alpha_{alpha}", f"{i}_fold_model.pickle"
            )
            self.save_model(reg, model_filepath)
            coef_filename = os.path.join(
                model_filepath, f"alpha_{alpha}", f"{i}_fold_coef.npy"
            )
            np.save(coef_filename, reg.coef_)
        # appending more lists to the metrics dictionary
        # from all folds, take the max_y
        metrics["max_y_test"] = max_y_test
        metrics["final_error_max_y_test"] = final_error_max_y_test
        metrics["molecule_max_error"] = molecule_max_error

        return metrics

    def perform_leave_p_out_cv(
        self, alpha, model_filepath, X, y, systems, p=1, n_splits=5
    ):
        """
        Performs Leave-P-Out cross-validation for model evaluation.

        Args:
            alpha (float): Regularization parameter for Lasso/Ridge regression
            model_filepath (str): Path to save model and results
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target values
            systems (list): List of system names/identifiers
            p (int): Number of samples to leave out in each iteration
            n_splits (int): Number of random splits to perform

        Returns:
            dict: Dictionary containing various metrics from cross-validation
        """
        metrics = {
            "train_mae": [],
            "test_mae": [],
            "all_mae": [],
            "train_mse": [],
            "test_mse": [],
            "all_mse": [],
            "max_y_test": [],
            "final_error_max_y_test": [],
            "molecule_max_error": [],
            "non_zero_parameters": [],
        }

        # Initialize scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Generate random splits
        n_samples = len(y)
        for split in range(n_splits):
            # Randomly shuffle indices
            indices = np.random.permutation(n_samples)

            # Split into chunks of size p
            for i in range(0, n_samples - p + 1, p):
                test_idx = indices[i : i + p]
                train_idx = np.array(
                    [idx for idx in range(n_samples) if idx not in test_idx]
                )

                # Split data
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                systems_test = [systems[i] for i in test_idx]

                # Initialize and fit model
                model = Lasso(alpha=alpha, max_iter=10000)
                model.fit(X_train, y_train)

                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                y_pred_all = model.predict(X_scaled)

                # Collect metrics
                collect_metrics(metrics, "train", y_train, y_pred_train)
                collect_metrics(metrics, "test", y_test, y_pred_test)
                collect_metrics(metrics, "all", y, y_pred_all)

                # Track maximum errors
                max_error_idx = np.argmax(np.abs(y_test - y_pred_test))
                metrics["max_y_test"].append(y_test[max_error_idx])
                metrics["final_error_max_y_test"].append(
                    np.abs(y_test[max_error_idx] - y_pred_test[max_error_idx])
                )
                metrics["molecule_max_error"].append(systems_test[max_error_idx])

                # Count non-zero parameters
                metrics["non_zero_parameters"].append(np.sum(model.coef_ != 0))

                # Save model for this fold
                model_name = f"model_split_{split}_chunk_{i//p}.pkl"
                save_model(model, os.path.join(model_filepath, model_name))

                # Save scaler
                scaler_name = f"scaler_split_{split}_chunk_{i//p}.pkl"
                save_model(scaler, os.path.join(model_filepath, scaler_name))

        return metrics

    def collect_metrics(self, metrics_dict, prefix, y_true, y_pred):
        metrics_dict[f"{prefix}_mae"].append(mean_absolute_error(y_true, y_pred))
        metrics_dict[f"{prefix}_mse"].append(mean_squared_error(y_true, y_pred))

    def save_model(self, model, filename):
        with open(filename, "wb") as file:
            pickle.dump(model, file)

    def log_max_error(self, alpha, metrics, log_filename):
        with open(log_filename, "a") as log_file:
            log_file.write(f"{alpha}\t")
            for i in range(len(metrics["max_y_test"])):
                log_file.write(
                    f"{i}\t{metrics['max_y_test'][i]}\t{metrics['final_error_max_y_test'][i]}\t{metrics['molecule_max_error'][i]}\t"
                )
            log_file.write("\n")

    def log_metrics(self, alpha, metrics, log_filename):
        with open(log_filename, "a") as log_file:
            log_file.write(
                f"{alpha}\t{np.mean(metrics['train_mae'])}\t{np.std(metrics['train_mae'])}\t"
                f"{np.min(metrics['train_mae'])}\t{np.max(metrics['train_mae'])}\t"
                f"{np.mean(metrics['test_mae'])}\t{np.std(metrics['test_mae'])}\t"
                f"{np.min(metrics['test_mae'])}\t{np.max(metrics['test_mae'])}\t"
                f"{np.mean(metrics['all_mae'])}\t{np.std(metrics['all_mae'])}\t"
                f"{np.min(metrics['all_mae'])}\t{np.max(metrics['all_mae'])}\t"
                f"{metrics['test_mae']}\t{metrics['non_zero_parameters']}\n"
            )

    def log_metrics_csv(self):
        """added for writing mae per model to a csv"""

        return NotImplementedError

    def log_results(self, filename, message):
        """Log results to a file."""
        with open(filename, "a") as f:
            f.write(message)

    def model_fitting(
        self, model_filepath, final_count_arr, target, systems, cross_val_type=None
    ):
        """Fit a LASSO model to the data and return the model and the predictions."""
        alpha_list = [10**exp for exp in range(-8, 3)]

        for alpha in alpha_list:
            start_time = time.time()
            alpha_path = os.path.join(model_filepath, f"alpha_{alpha}")
            os.makedirs(alpha_path, exist_ok=True)
            print(f"==== Training model with alpha = {alpha} ====")
            metrics = self.perform_cross_validation(
                alpha, model_filepath, final_count_arr, target, systems
            )
            self.log_metrics(
                alpha, metrics, os.path.join(model_filepath, f"{alpha}_overall_log.txt")
            )
            self.log_max_error(
                alpha,
                metrics,
                os.path.join(model_filepath, f"{alpha}_max_error_log.txt"),
            )
            end_time = time.time()
            print(f"Time taken for alpha = {alpha}: {end_time - start_time} seconds")

        # TODO - perform leave one out p cross val and log those metrics for comparison, wiot cross val type

        return


    def main(
        self,
        mcsh,
        rcut,
        overall_sig,
        cutoff_sig,
        ccsdt_file,
        pbe_file,
        atomic_number_file,
        count_path,
    ):
        # Load energy and atomic number data
        ccsdt_energy = self.load_json(ccsdt_file)
        pbe_energy = self.load_json(pbe_file)
        atomic_number_dict = self.load_json(atomic_number_file)

        # computes the difference in formation energies between CCSDT and PBE, this is the target we aim to train on
        target_dict = self.calculate_target_variable(
            ccsdt_energy, pbe_energy, atomic_number_dict
        )

        # systems = list(target_dict.keys())
        print(np.mean(abs(np.array(list(target_dict.values())))))
        count_file = os.path.join(
            count_path,
            f"mcsh_{mcsh}_rcut_{rcut}.0",
            f"count_array_overall_{overall_sig}_system_{cutoff_sig}.csv",
        )  # in newest count_path, just make sure this matches the name of the data files
        # these are tagged with the mcsh and rcut settings as well
        final_count_arr, target, systems = self.sort_count_array(
            count_file, target_dict
        )

        # call the function for LASSO regression
        self.model_fitting(self.model_filepath, final_count_arr, target, systems)
        return

if __name__ == "__main__":
    ccsdt_file = "ccsdt_energy.json"
    pbe_file = "pbe_energy.json"
    atomic_number_file = "atoms_count_mat.json"
    # count_path = "/storage/home/hcoda1/0/ssahoo41/cedar_storage/ssahoo41/exact_exchange_work/NNS_subsampling/partitioning_scheme/n_vac_csv"
    count_path = os.path.join(
        "..", "partitioning_scheme", "new_data"
    )  # changed from n_vac_csv on 10/20/2024
    # the count_path is the location of the count arrays after data transformation
    wd_path = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)  # make current

    if len(sys.argv) < 3:
        print("Usage: python script.py <overall_sig> <cutoff_sig>")
        print(sys.argv)
        sys.exit(1)

    overall_sig = float(sys.argv[1])
    sys_sig = float(sys.argv[2])
    stdscale = sys.argv[3]
    mcsh = 2  # change to mcsh tested
    rcut = 2  # change to rcut tested

    # make the Energy object to call main on it
    e = EnergyCorrectionFitter(
        mcsh=mcsh,
        rcut=rcut,
        overall_sig=overall_sig,  # overall cutoff sig for recording
        cutoff_sig=sys_sig,  # system specific cutoff sig
        ccsdt_file=ccsdt_file,  # file with CCSDT (highly accurate) energies
        pbe_file=pbe_file,
        atomic_number_file=atomic_number_file,
        count_path=count_path,  # path with latest matrices of molecule counts, after data cleaning and subsampling
        stdscale=stdscale,  # True or False depending on whether the data used a standard scaler, for logging
    )

    # Execute the main function with the specified arguments
    e.main(
        mcsh,
        rcut,
        overall_sig,
        sys_sig,
        ccsdt_file,
        pbe_file,
        atomic_number_file,
        count_path,
    )


def calculate_formation_energy(energy_dict, atoms_count_array):
    """Calculates formation energy from energy dictionary and atoms count array."""

    assert len(energy_dict) == atoms_count_array.shape[0], "Mismatch in lengths of energy_dict and atoms_count_array"

    energy_array = np.array(list(energy_dict.values())) * Ha_to_eV
    molecules = list(energy_dict.keys())
    reg = LinearRegression(fit_intercept=False)
    reg.fit(atoms_count_array, energy_array)
    predicted_energy = reg.predict(atoms_count_array)
    formation_energy = energy_array - predicted_energy
    return {molecule: energy for molecule, energy in zip(molecules, formation_energy)}

def calculate_target_variable(ccsdt_energy, pbe_energy, atomic_number_dict):
    """Calculates target variable for model training."""
    ccsdt_formation_en = calculate_formation_energy(ccsdt_energy, np.array(list(atomic_number_dict.values())))
    pbe_formation_en = calculate_formation_energy(pbe_energy, np.array(list(atomic_number_dict.values())))

    target_dict = {key: ccsdt_formation_en[key] - pbe_formation_en.get(key, 0) for key in ccsdt_formation_en}

    return target_dict

def sort_count_array(count_file, target_dict):
    """Sorts count array based on the order of molecules in CCSDT formation energy dictionary."""
    df = pd.read_csv(count_file, header=None)
    target_dict = {key: target_dict[key] for key in target_dict if key in df[1].values} # Filter out molecules not in target_dict

    df['sort_order'] = df[1].map(lambda x: list(target_dict.keys()).index(x) if x in target_dict else None)
    df_sorted = df.sort_values(by='sort_order').iloc[:, 2:].drop(columns=['sort_order'])
    return df_sorted.to_numpy(), np.array(list(target_dict.values())), list(target_dict.keys())

def perform_cross_validation(alpha, model_filepath, final_count_arr, target, systems):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {
        'train_mae': [], 'test_mae': [], 'all_mae': [],
        'train_mse': [], 'test_mse': [], 'all_mse': [],
        'non_zero_parameters': []
    }
    # we will append these lists with the maximum error in each fold
    max_y_test = []
    final_error_max_y_test = []
    molecule_max_error = []

    for i, (train_index, test_index) in enumerate(kf.split(systems)):
        X_train, X_test = final_count_arr[train_index], final_count_arr[test_index]
        # scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        y_train, y_test = target[train_index], target[test_index]
        final_count_arr_scaled = scaler.transform(final_count_arr)

        # Fit the LASSO model
        reg = Lasso(alpha=alpha, fit_intercept=False, max_iter=20000, selection='random')
        reg.fit(X_train, y_train)
        # Check if Lasso converged
        if reg.n_iter_ == reg.max_iter:
            print(f"Warning: Lasso did not converge for Alpha = {alpha}, Fold {i}. Reached maximum iterations: {reg.n_iter_}")
        print(f"Alpha = {alpha}, Fold {i}: Number of iterations = {reg.n_iter_}")
        y_test_pred = reg.predict(X_test)
        max_test_idx = np.argmax(abs(y_test))
        max_y_test.append(y_test[max_test_idx])
        final_error_max_y_test.append(y_test[max_test_idx] - y_test_pred[max_test_idx])
        molecule_max_error.append(systems[test_index[max_test_idx]])

        #final_count_arr_scaled = scaler.transform(final_count_arr)
        # Collect and store metrics
        collect_metrics(metrics, 'train', y_train, reg.predict(X_train))
        # save y_train and y_train_pred to a file in the folder for alpha for each fold

        y_train_pred = reg.predict(X_train)
        train_pred_filename = os.path.join(model_filepath, f"alpha_{alpha}", f"{i}_fold_train_pred.npy")
        np.save(train_pred_filename, y_train_pred)
        train_true_filename = os.path.join(model_filepath, f"alpha_{alpha}", f"{i}_fold_train_true.npy")
        np.save(train_true_filename, y_train)

        collect_metrics(metrics, 'test', y_test, reg.predict(X_test))
        collect_metrics(metrics, 'all', target, reg.predict(final_count_arr_scaled))
        
        # Save the model
        model_filename = os.path.join(model_filepath, f"alpha_{alpha}", f"{i}_fold_model.pickle")
        save_model(reg, model_filename)
        coef_filename = os.path.join(model_filepath, f"alpha_{alpha}", f"{i}_fold_coef.npy")
        np.save(coef_filename, reg.coef_)
    # appending more lists to the metrics dictionary
    metrics['max_y_test'] = max_y_test
    metrics['final_error_max_y_test'] = final_error_max_y_test
    metrics['molecule_max_error'] = molecule_max_error
    metrics['non_zero_parameters'].append(np.count_nonzero(reg.coef_))

    return metrics

def collect_metrics(metrics_dict, prefix, y_true, y_pred):
    metrics_dict[f'{prefix}_mae'].append(mean_absolute_error(y_true, y_pred))
    metrics_dict[f'{prefix}_mse'].append(mean_squared_error(y_true, y_pred))

def save_model(model, filename):
    with open(filename, "wb") as file:
        pickle.dump(model, file)

def log_max_error(alpha, metrics, log_filename):
    with open(log_filename, 'a') as log_file:
        log_file.write(f"{alpha}\t")
        for i in range(len(metrics['max_y_test'])):
            log_file.write(f"{i}\t{metrics['max_y_test'][i]}\t{metrics['final_error_max_y_test'][i]}\t{metrics['molecule_max_error'][i]}\t")
        log_file.write("\n")

def log_metrics(alpha, metrics, log_filename):
    # print train mae
    print(metrics['train_mae'])
    with open(log_filename, 'a') as log_file:
        log_file.write(f"{alpha}\t{np.mean(metrics['train_mae'])}\t{np.std(metrics['train_mae'])}\t"
                       f"{np.min(metrics['train_mae'])}\t{np.max(metrics['train_mae'])}\t"
                       f"{np.mean(metrics['test_mae'])}\t{np.std(metrics['test_mae'])}\t"
                       f"{np.min(metrics['test_mae'])}\t{np.max(metrics['test_mae'])}\t"
                       f"{np.mean(metrics['all_mae'])}\t{np.std(metrics['all_mae'])}\t"
                       f"{np.min(metrics['all_mae'])}\t{np.max(metrics['all_mae'])}\t"
                       f"{metrics['test_mae']}\t{metrics['non_zero_parameters']}\n")

def log_results(filename, message):
    """Log results to a file."""
    with open(filename, 'a') as f:
        f.write(message)

def model_fitting(model_filepath, final_count_arr, target, systems):
    """ Fit a LASSO model to the data and return the model and the predictions."""
    # Generate alpha values for fine-tuning around 1e-3
    #alpha_list = [10**exp for exp in range(-4, -2, 1)]
    #alpha_list += [1e-3 + i*(1e-4) for i in range(-5, 6)]  # Adding more granularity around 1e-3
    alpha_list = [10**exp for exp in range(-8,3)] 
    for alpha in alpha_list:
        start_time = time.time()
        alpha_path = os.path.join(model_filepath, f"alpha_{alpha}")
        os.makedirs(alpha_path, exist_ok=True)
        print(f"==== Training model with alpha = {alpha} ====")
        metrics = perform_cross_validation(alpha, model_filepath, final_count_arr, target, systems)
        log_metrics(alpha, metrics, os.path.join(model_filepath, "overall_log.txt"))
        log_max_error(alpha, metrics, os.path.join(model_filepath, "max_error_log.txt"))
        end_time = time.time()
        print(f"Time taken for alpha = {alpha}: {end_time - start_time} seconds")
    return

def main(overall_sig, cutoff_sig, ccsdt_file, pbe_file, atomic_number_file, count_path):
    # Load energy and atomic number data
    ccsdt_energy = load_json(ccsdt_file)
    pbe_energy = load_json(pbe_file)
    atomic_number_dict = load_json(atomic_number_file)
    target_dict = calculate_target_variable(ccsdt_energy, pbe_energy, atomic_number_dict)
    #systems = list(target_dict.keys())
    print(f"Baseline MAE between PBE and CCSD(T) = {np.mean(abs(np.array(list(target_dict.values()))))}")
    count_file = os.path.join(count_path, f"count_array_overall_{overall_sig}_system_{cutoff_sig}.csv")
    final_count_arr, target, systems = sort_count_array(count_file, target_dict)
    # call the function for LASSO regression
    model_fitting(lasso_model_filepath, final_count_arr, target, systems)
    return

# Script Execution Entry Point
if __name__ == "__main__":
    ccsdt_file = "ccsdt_energy.json"
    pbe_file = "pbe_energy.json"
    atomic_number_file = "atoms_count_mat.json"

    mcsh_order = int(sys.argv[1])
    rcut = float(sys.argv[2])
    overall_sig = float(sys.argv[3])
    sys_sig = float(sys.argv[4])
    print(overall_sig, sys_sig)
    base_path = "/storage/home/hcoda1/0/ssahoo41/cedar_storage/ssahoo41/exact_exchange_work/thesis_datagen/publication_purpose/subsampling_script/partitioning"
    count_path = os.path.join(base_path, f"mcsh_{mcsh_order}_rcut_{rcut}")
    print(f"Count path: {count_path}")
    lasso_model_filepath = os.path.join("models_lasso_pbe_shuffle_True", f"mcsh_{mcsh_order}_rcut_{rcut}", f"model_all_{overall_sig}_sys_{sys_sig}_lasso")
    os.makedirs(lasso_model_filepath, exist_ok=True)
    # Execute the main function with the specified arguments
    main(overall_sig, sys_sig, ccsdt_file, pbe_file, atomic_number_file, count_path)