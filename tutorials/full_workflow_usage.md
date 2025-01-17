
# Full Workflow Tutorial

This is a full tutorial if you wish to replicate the process for training the correction scheme model. For applications of the model, please use the [Quick Start](tutorials/quick_start.md) guide instead. 

First, you must have generated molecule data using the latest experimental version of SPARC, dev-SPARC, with multipole features, linked here:

The outputs of this process will look like this:

Please load those into the data/molecules folder, where each molecule is its own folder (this should match the outputs of dev SPARC)

First, install the package:

## Installation

Clone the repo:

```bash
   git clone https://github.com/yourusername/CORRECTION_SCHEME.git
```
Set the location of the molecule or system folder with:

```bash
export MOLECULES_DATA_PATH="/path/to/molecules"

```
then run the workflow with 

```bash
python workflow.py

```

Note that any part of the workflow can be run separately, but the workflow module will run each step for you, given an initial molecules folder and minor configuration. 