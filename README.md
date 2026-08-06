# Phylogenetic-Imputation
Code and data for the imputing species trait data using Tabular prior fitted networks, and making comparisons to RPhylopars and multiple imputation by chained equations using random forest.

In addition to the files used within our study, we also provide the following tutorial for using TabPFN for imputing missing species trait data:

## Prerequisites 

### Installing TabPFN
TabPFN is currently available only on Python. See the [official TabPFN GitHub](https://github.com/PriorLabs/TabPFN) for the installation process and their Quick Start guide.

### CUDA
Running many deep learning models is best done using your computed GPUs rather than CPUs. CUDA is a platform for allowing this, and is recommended to get the best out of TabPFN. A guide for getting set up with CUDA can be found [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).

## TabPFN for trait data imputation

Our tutorial is based on training TabPFN purely on complete cases within an empirical trait dataset. You will first need the following packages.
```
import pandas as pd
import math

import numpy as np
import pickle
import torch
import gc
import os

from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
import Preprocessing as process_Data
```


```
Imputing species trait data is simple:
- Preprocess your data
- Identify complete cases
- Define and fit your TabPFN model
- Use this to impute your full dataset
- Use prior imputations to inform future ones (including in the trait process)
- Save your multiply imputed datasets
- Use each for your analyses and pool using Rubin's' Rules
```






