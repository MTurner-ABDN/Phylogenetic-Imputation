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

current_rds_path <- paste0("Path to your tree/trait .rds file here", sep="")
current_rds <- readRDS(current_rds_path)

# Load current tree and amputed trait data
Tree <- current_rds[[2]]
TreeGraph = Phylo2DirectedGraph(Tree)
A <- current_rds[[1]]

trait_cols <- "List of your trait names here" 
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
    to_save_filename <- paste0("Path to save your phylogenetic eigenvectors .rds file", sep="")
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
    to_save_filename <- paste0("Path to save your phylogenetic eigenvectors .rds file", sep="")
    saveRDS(BuildTrimmedPEM, file = to_save_filename)

    return(NA)
  })
```
2) Combine your trait data and normalized phylogenetic eigenvectors:
```
library("MPSEM")
normalize <- function(x){
  return((x - min(x))/ (max(x) - min(x)))
}

curr_eigens_location <- paste0("Path to your eigenvector .rds file here", sep="")
curr_eigens <- readRDS(file = curr_eigens_location)
current_eigen_data <- as.data.frame(curr_eigens$u)
normalized_eigens <- apply(current_eigen_data, 2, normalize)

# Load corresponding amputed traits
curr_traits_location <- paste0("Path to your tree/trait .rds file here", sep="")
curr_traits <- readRDS(file = curr_traits_location)
curr_traits <- curr_traits[[1]][, c(2,3,4,5,...)]  # Select just your trait data

combined_traits_eigens <- cbind(curr_traits, normalized_eigens)  # Combine 

out_loc <- paste0("Path to your save location here", sep="")
write.table(x = combined_traits_eigens, file = out_loc, row.names = FALSE, col.names = TRUE, sep = "\t")

```

### Trait data imputation
#### Prerequisites 

##### Installing TabPFN
TabPFN is currently available only on Python. See the [official TabPFN GitHub](https://github.com/PriorLabs/TabPFN) for the installation process and their Quick Start guide.

##### CUDA
Running many deep learning models is best done using your computed GPUs rather than CPUs. CUDA is a platform for allowing this, and is recommended to get the best out of TabPFN. A guide for getting set up with CUDA can be found [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).

3)

Imputing species trait data is simple:
- Load trait data and eigenvectors
- Identify complete cases
- Define and fit your TabPFN model
- Use this to impute your full dataset
- Use prior imputations to inform future ones (including in the trait process)
- Save your multiply imputed datasets
- Use each for your analyses and pool using Rubin's' Rules


We then preprocess our dataset, including 




