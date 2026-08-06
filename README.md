# Phylogenetic-Imputation
Code and data for the imputing species trait data using Tabular prior fitted networks, and making comparisons to RPhylopars and multiple imputation by chained equations using random forest.

In addition to the files used within our study, we also provide the following tutorial for using TabPFN for imputing missing species trait data:

## Prerequisites 

### Installing TabPFN
TabPFN is currently available only on Python. See the [official TabPFN GitHub](https://github.com/PriorLabs/TabPFN) for the installation process and their Quick Start guide.

### CUDA
Running many deep learning models is best done using your computed GPUs rather than CPUs. CUDA is a platform for allowing this, and is recommended to get the best out of TabPFN. A guide for getting set up with CUDA can be found [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).

## TabPFN for trait data imputation

### Preprocessing
1) With a .rds file of your trait data and accompanying phylogenetic tree compute trait specific phylogenetic eigenvectors using:
```
require(Rphylopars)
require(phytools)
require(norm)
require(caper)
require(rowr)
require(plyr)
require(dplyr)
require(mice)
require(MPSEM)
require(tidyr)
require(ggplot2)
require(ggridges)
require(ggpubr)
require(progress)
require(randomForest)
require(lme4)
require(lsmeans)
require(lmerTest)
require(stringr)
require(xlsx)
require(arm)
library(pdftools)

current_rds_path <- paste0("Path_to_tree_trait.rds", sep="")
current_rds <- readRDS(current_rds_path)

# Load current tree and amputed trait data
Tree <- current_rds[[2]]
TreeGraph = Phylo2DirectedGraph(Tree)
A <- current_rds[[1]]

trait_cols <- c("V2","V3","V4","V5") # List of your trait names here
trait_stats <- sapply(A[trait_cols], function(x) sum(is.finite(x)))
most_complete_trait <- names(trait_stats)[which.max(trait_stats)]

for (trait in traits) {
  curr_trait <- trait
  print("Working on next pass: new trait")
  #if (sum(is.finite(A[[trait]])) > 10) {  # Ensure current trait has enough values for PEM

  old <- options(warn = 2) 

  tryCatch({
    TrimmedRows = A[complete.cases(A[[curr_trait]]),]  

    spmatch = match(Tree$tip.label, TrimmedRows$TreeTips)  # Match tips with tree
    TrimmedTree = drop.tip(Tree, Tree$tip.label[is.na(spmatch)])  # Drop tips that don't match from tree

    TrimmedTreeGraph = Phylo2DirectedGraph(TrimmedTree)  # create graph
    print("Working on next pass: Starting tryCatch process")
    TrimmedPEM = RobustPEM(TrimmedRows[[curr_trait]], TrimmedTreeGraph)  # Pass to Robust PEM function - i.e. build PEM using fitSimple
    ErrorPresent = TrimmedPEM[[2]]  # Get whether error is present
    TrimmedPEM = TrimmedPEM[[1]]  # Get trimmed PEM

    BuildTrimmedPEM = PEM.build(TreeGraph, d = "distance", sp = "species", a = TrimmedPEM$a, psi = TrimmedPEM$psi) # Build trimmed PEM with full TreeGraph but with s and psi from TrimmedPEM
    TrimmedPEM_DF = as.data.frame(BuildTrimmedPEM)
    to_save_filename <- paste0("Save_path_for_your_phylogenetic_eigenvectors.rds", sep="")
    saveRDS(BuildTrimmedPEM, file = to_save_filename)
  }, error = function(cond){
    print("Error occurred")
    TrimmedRows = A[complete.cases(A[[most_complete_trait]]),]  

    spmatch = match(Tree$tip.label, TrimmedRows$TreeTips)  # Match tips with tree

    TrimmedTree = drop.tip(Tree, Tree$tip.label[is.na(spmatch)])  # Drop tips that don't match from tree
    TrimmedTreeGraph = Phylo2DirectedGraph(TrimmedTree)  # create graph

    TrimmedPEM = RobustPEM(TrimmedRows[[most_complete_trait]], TrimmedTreeGraph)  # Pass to Robust PEM function - i.e. build PEM using fitSimple
    ErrorPresent = TrimmedPEM[[2]]  # Get whether error is present
    TrimmedPEM = TrimmedPEM[[1]]  # Get trimmed PEM

    BuildTrimmedPEM = PEM.build(TreeGraph, d = "distance", sp = "species", a = TrimmedPEM$a, psi = TrimmedPEM$psi) # Build trimmed PEM with full TreeGraph but with s and psi from TrimmedPEM
    TrimmedPEM_DF = as.data.frame(BuildTrimmedPEM)
    to_save_filename <- paste0("Save_path_for_your_phylogenetic_eigenvectors.rds", sep="")
    saveRDS(BuildTrimmedPEM, file = to_save_filename)

    return(NA)
  })
}
```
2) Combine your trait data and normalized phylogenetic eigenvectors:
```
library("MPSEM")

normalize <- function(x){
  return((x - min(x))/ (max(x) - min(x)))
}

for (trait in traits) {
  curr_eigens_location <- paste0("Save_path_for_your_phylogenetic_eigenvectors_for_the_current_trait.rds", sep="")
  curr_eigens <- readRDS(file = curr_eigens_location)
  current_eigen_data <- as.data.frame(curr_eigens$u)
  normalized_eigens <- apply(current_eigen_data, 2, normalize)
  
  # Load corresponding amputed traits
  curr_traits_location <- paste0("Path_to_tree_trait.rds", sep="")
  curr_traits <- readRDS(file = curr_traits_location)
  curr_traits <- curr_traits[[1]][, c(2,3,4,5,...)]  # Select just your trait data
  
  combined_traits_eigens <- cbind(curr_traits, normalized_eigens)  # Combine 
  
  out_loc <- paste0("Path_to_your_trait_eigens_save_location_for_the_current_trait.txt", sep="")
  write.table(x = combined_traits_eigens, file = out_loc, row.names = FALSE, col.names = TRUE, sep = "\t")
}
```

### Trait data imputation
#### Prerequisites 
##### Installing TabPFN
TabPFN is currently available only on Python. See the [official TabPFN GitHub](https://github.com/PriorLabs/TabPFN) for the installation process and their Quick Start guide.

##### CUDA
Running many deep learning models is best done using your computed GPUs rather than CPUs. CUDA is a platform for allowing this, and is recommended to get the best out of TabPFN. A guide for getting set up with CUDA can be found [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).

With TabPFN and CUA installed, to impute trait data using TabPFN:
1) Load packages and define function for identifying complete cases
```
import pandas as pd
import math

import numpy as np
import gc
import os

from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
import Preprocessing as process_Data

def get_CCs(amputed_data, trait):
    complete_cases = []
    # Get a mask of observed/not from amputed:
    curr_feature = amputed_data[:, trait]
    curr_feat_bool = np.isnan(curr_feature)

    for idx, missing in enumerate(curr_feat_bool):
        if not missing:
            complete_cases.append(amputed_data[idx, :])

    complete_cases = np.asarray(complete_cases)
    print(f"Training size for feature {trait} :  {len(complete_cases)}")

    return complete_cases
```

2) Data preprocessing:
```
# Load out dataset
Amputed_dataset = np.genfromtxt(f"Location_of_your_trait_data.txt", skip_header=True)

# Make a copy to be filled with imputed data - for updating 
to_be_filled = Amputed_dataset[:, :n].copy()

# Identify missing indexes per feature (trait)
missing_indexes = process_Data.get_missing_indexes_nan(Amputed_dataset[:, :n].copy(), group=group)  # We get the missing indexes
f1, ... fn = process_Data.split_indexes_by_feature(missing_indexes, group=group)  # Split by feature
features = [f1, ... fn]

# Define number of multiply imputed datasets to create
Number_of_multiply_imputed_datasets = 10

# Define a dictionary for saving multiply imputed datasets
to_be_filled_dict = {}
for t in range(Number_of_multiply_imputed_datasets):
    to_be_filled_dict[f"to_be_filled_{t}"] = to_be_filled.copy()
```

3) Train and inference loop:
```
imputed_feat_list = []
# Looping over each trait
for feat in range(0, n, 1):

    imputed_feat_list.append(feat)
    data = np.genfromtxt(f"Path_to_corresponding_trait_eigenvector_data_for_trait_{feat + 2}.txt", delimiter="\t")
    data = data[1:, 1:]

    # Include previously imputed data in our current training
    if n != 0:  # Excluding the first pass
        for so_far_imputed in imputed_feat_list:
            data[:, so_far_imputed] = to_be_filled[:, so_far_imputed]

    # Generate binary mask of observed/missing data locations
    curr_traits = data[:, :n]
    curr_eigens = data[:, n:]
    curr_mask = process_Data.get_nan_mask_matrix(curr_traits.copy())
    data = np.concatenate((curr_traits, curr_mask, curr_eigens), axis=1)
    data = data[:, :500]
    # Identify complete cases for the current trait
    data = get_CCs(data, feat)

    # Our xs are our predictors (non-focal traits, missingness mask, and eigenvectors), and our ys are our targets (our focal trait)
    # Remove focal trail from xs
    X_train = np.delete(data.copy(), feat, 1)
    y_train = data[:, feat].copy()  # This worked fine

    # Define the TabPFN model
    reg = TabPFNRegressor(device="cuda", n_estimators=30).create_default_for_version(ModelVersion.V2)
    # Fit the model to our training data
    reg.fit(X_train, y_train)

    # Load data for inference of missing data
    test_xs = np.genfromtxt(f"Path_to_corresponding_trait_eigenvector_data_for_trait_{feat + 2}.txt", delimiter="\t")  # Load the amputed data and process
    test_xs = test_xs[1:, 1:]

    # Generate mask of missing and observed data
    curr_traits = test_xs[:, :n]
    curr_eigens = test_xs[:, n:]
    curr_mask = process_Data.get_nan_mask_matrix(curr_traits.copy())
    test_xs = np.concatenate((curr_traits, curr_mask, curr_eigens), axis=1)
    test_xs = test_xs[:, :500]

    # Include previously imputed data in our current inference
    if n != 0:  # Excluding the first pass
        for so_far_imputed in imputed_feat_list:
            test_xs[:, so_far_imputed] = to_be_filled[:, so_far_imputed]

    # Remove focal trail from xs
    X_Test_copy = np.delete(test_xs.copy(), feat, 1)  # Remove focal feature
    print(f'Test shape: {np.shape(X_Test_copy)}, feature missing: {feat}')

    # Make predictions provided our non-focal traits, missingness mask, and eigenvectors 
    temp_predictions = reg.predict(X_Test_copy)  # First predictions for use in future runs
    quantiles = np.random.beta(5, 5, Number_of_multiply_imputed_datasets)  # Define quantiles

    # Then predictions for our multiply imputed datasets 
    quantile_predictions = reg.predict(X_Test_copy, output_type="quantiles", quantiles=quantiles)  # Get quantile predictions
    quantile_predictions = np.asarray(quantile_predictions).T

    # Update our dictionaries of current quantile predictions, and to_be_filled multiply imputed dataset dictionary
    quant_pred_dict = {}
    for t in range(Number_of_multiply_imputed_datasets):  # Increment t
        quant_pred_dict[f"Quant_pred_{t}"] = quantile_predictions[:, t]  # Dictionary of quantile predictions for current feature.

    for t in range(Number_of_multiply_imputed_datasets):  # For dictionary slot t
        for s, f in features[feat]:  # Get the position to be replaced in this focal feature
            to_be_filled_dict[f"to_be_filled_{t}"][s][f] = quant_pred_dict[f"Quant_pred_{t}"][s]

    # Update our temp to_be_filled dataset
    for s, f in features[feat]:  # For each missing value from the given feature in X_test
        to_be_filled[s][f] = temp_predictions[s]  ######### Here we update test_xs instead so that the next iteration has access to these imputations
```

4) Save multiply imputed datasets
```
for t in range(Number_of_multiply_imputed_datasets):
    np.savetxt(fname=f"Save_location_for_your_multiply_imputed_dataset_{t}.csv", X=np.asarray(filled_arrays_dictionary[f"to_be_filled_{t}"]), delimiter="\t")
```

5) Conduct your analysis with each multiply imputed datasets and pool the resulting parameter/coefficient estimates using Rubin's rules




