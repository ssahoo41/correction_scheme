# Usage and Quick Start

This guide is intended to get users up and running using the best model from the paper to correct system energies. 
<!-- TODO: insert name of paper once on ArXiv -->

Note that using the correction scheme in its entirety will allow you to train correction models from scratch based on molecular data. This guide can be found under `tutorials/full_workflow_usage`

For this quick start, the flow will be: 

![](assets/2024-12-18-15-09-34.png)

1. First, load your DFT output energies onto your system
2. import the package as `import correction_scheme as corr_scheme` 
3. Change the path to the energies to your relative path from the folder in which you are using the package
4. Call on package.predict_with_best_model()
5. use the output of the function as the corrected DFT energies 



