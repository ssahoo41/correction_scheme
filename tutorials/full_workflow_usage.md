
# Full Workflow Tutorial

This is a full tutorial if you wish to replicate the **full process for training the correction scheme model** on featurized energies from DFT. 

For applications of the model, please use the [Quick Start](tutorials/quick_start.md) guide instead. 

First, you must have generated molecule data using the development version of [SPARC](https://github.com/ssahoo41/dev_SPARC_PBEq) with HSMP/multipole features implemented.

The outputs of this process will look like this:



Please load those into a `molecules` folder of your choosing, where each molecule is its own folder (this should match the outputs of dev_SPARC)

First, install the package:

## Installation

Clone the repo:

```bash
   git clone https://github.com/ssahoo41/correction_scheme/tree/tutorials
```

## Workflow Replication

Set the location of the molecule or system folder with:

```bash
export MOLECULES_DATA_PATH="/path/to/molecules"

```
Begin the pre-defined workflow with:

```bash
python workflow.py
```
>Note: please use the python alias on your system for python >= 3.9.

>Note: part of the workflow can be run separately, but the workflow module will run each step for you, given an initial molecules folder and minor configuration. 


