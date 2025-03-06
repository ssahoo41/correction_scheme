# Electronic Structure Correction Scheme

A Python package that improves energies from Density Functional Theory (DFT) calculations using electronic environment descriptors and machine learning techniques. 

This repo contains:

- Pre-trained models to correct DFT energies to high-accuracy CCSD(T) level
    - NOTE: final models undergoing additional training
- Complete training workflow for creating custom correction models
- [SPARC DFT](https://github.com/ssahoo41/dev_SPARC_PBEq) with multipole features for descriptor generation

## Quick Start

For those who want to apply pre-trained models to their own systems:

<!-- TODO once we have the best model incorporate that part of the process-->

```bash
# Set your data path as environment variable
export PREDICT_ENERGIES_MOLECULES_DATA_PATH="/path/to/molecules"
# these should be the molecules you want to predict energies for

# Run prediction with pre-trained model
# dot notation should work if run from project root
python -m correction_scheme.predict
```

For detailed usage instructions, see the [Quick Start Guide](./tutorials/quick_start.md).

## Full Workflow

To train your own electron density-based correction scheme model:

1. Generate descriptors using the [SPARC DFT code with multipole features](https://github.com/ssahoo41/dev_SPARC_PBEq)
2. Follow our [Full Workflow Tutorial](./tutorials/full_workflow_usage.md)

## Data Generation

Descriptor data can be generated using the development version of SPARC with HSMP/multipole features implemented:

- [How to Generate Fingerprinted Data Using SPARC](./tutorials/dev-SPARC-data-gen.md)

## Installation of Correction Scheme Package

```bash
# Clone the repository
git clone https://github.com/ssahoo41/correction_scheme.git
cd correction_scheme

# Install the package and dependencies
pip install -e .
```


## Requirements

- Python >= 3.9
- NumPy, Pandas, SciPy, scikit-learn
- ASE (Atomic Simulation Environment)
- See requirements.txt for full dependencies

## Citation

If you use this software in your research, please cite:
<!-- Add citation information when available -->

## Authors

- Sushree Jagriti Sahoo (ssahoo41@gatech.edu)
- Lisette del Pino (lpino3@gatech.edu)

## License

[MIT License](./LICENSE)
<!-- 
# Electronic Structure Correction Scheme

motivations

A Python package to improve energies from the Density Functional Theory based on electronics environments and machine learning techniques. 

Please read the [Quick Start](tutorials/quick_start.md) to get quickly started with the pre-trained models. 

We have also made the training workflow available. Please read the Full Workflow Tutorial [Full Workflow Tutorial](tutorials/full_workflow_usage.md) to use the workflow to train your electron density-based correction scheme model. 

The descriptor data can be generated using [SPARC](https://github.com/ssahoo41/dev_SPARC_PBEq). 

https://pmc.ncbi.nlm.nih.gov/articles/PMC3928866/


 -->

