# Usage and Quick Start

This guide is intended to get users up and running using the best model from the paper to correct formation energies for molecules and adsorption energies for catalytic systems. 
<!-- TODO: insert name of paper once on ArXiv -->

Note that using the correction scheme in its entirety will allow you to train electron density-based correction models from scratch based on molecular data. This guide can be found under [Full Workflow Usage](tutorials/full_workflow_usage.md)

For this quick start, the flow will be: 

![](assets/2024-12-18-15-09-34.png)


Step 0: 

Generate the electron density descriptors and DFT total energies for systems using SPARC multipole features branch. 
- The outputs of the DFT calculator SPARC with multipole features are stored in the format: `HSMP_l_{l_value}_rcut_{rcut_value}_spin_typ_0.csv`
  - If your output file has a different name, assure you are using the multipole features branch and that the output of your file is a `.csv`
1. If the name of your output file is named EXAMPLE, 
   1. Note that the structure of the data must be a dataframe, with molecules or systems corresponding to different rows


Step 1: 

- You must set the correct data path for the molecules folder with the following in bash or command line:

```bash
export MOLECULES_DATA_PATH="/path/to/molecules"
```
This is the path that contains the output files of SPARC DFT calculations. 

then start the workflow with: 

```bash
python workflow.py
```

> requirements.txt
> init.py 

```python3

```


1. Change the path to the energies to your relative path from the folder in which you are using the package
2. Call on package.predict_with_best_model()
3. use the output of the function as the corrected DFT energies


---
<!-- 
TODO

1. add data preparation folder (from SPARC to h5 data)
   1. preparation should also be a class
2. convert discard vacuum into class
3. convert overall and system subsample into class 
   1. overall subsample is reading from pkl file so take this into consideration 
4. partition class
5. review and remove redundancies from model fitting and  correction scheme fitting 
6. utilities module/python file
7. finish tutorials
   1. in tutorials provide examples of how to validate the code
   2. tutorials should just be a jupyter notebook 
8. turn module into an installable package (use init.py method)
9. meet on Friday  -->
  
