# Phylogenetic-Imputation
Code and data for the imputing species trait data using Tabular prior fitted networks, and making comparisons to RPhylopars and multiple imputation by chained equations using random forest.

In addition to the files used within our study, we also provide the following tutorial for using TabPFN for imputing missing species trait data:

## Prerequisites 

### Installing TabPFN
TabPFN is currently available only on Python. See the [official TabPFN GitHub](https://github.com/PriorLabs/TabPFN) for the installation process and their Quick Start guide.

### CUDA
Running many deep learning models is best done using your computed GPUs rather than CPUs. CUDA is a platform for allowing this, and is recommended to get the best out of TabPFN. A guide for getting set up with CUDA can be found [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).

## TabPFN for trait data imputation
1) With your trait data and accompanying phylogenetic tree compute trait specific phylogenetic eigenvectors using:
```
current_rds_path <- paste0("Updated_rds/",group,"/",scenario ,"/Missing_",missingness,"/Simulated_",group,"_Seed_",seed,"_",num,"_",scenario,"_Missing_",missingness,".rds", sep="")
current_rds <- readRDS(current_rds_path)

# Load current tree and amputed trait data
Tree <- current_rds[[2]]
TreeGraph = Phylo2DirectedGraph(Tree)
A <- current_rds[[1]]

trait_cols <- names(A)[grepl("^V", names(A)) & names(A) != "V1"]  # Find which trait is most complete.
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
    to_save_filename <- paste0("../Data/TemporaryData/",group,"/Eigens/",scenario,"/Missing_",missingness ,"/Trimmed_",group,"_Eigens_Seed_",seed,"_",num,"_",scenario,"_missing_",missingness,"_trait_",trait,".rds", sep="")
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
    to_save_filename <- paste0("../Data/TemporaryData/",group,"/Eigens/",scenario,"/Missing_",missingness ,"/Trimmed_",group,"_Eigens_Seed_",seed,"_",num,"_",scenario,"_missing_",missingness,"_trait_",trait,".rds", sep="")
    saveRDS(BuildTrimmedPEM, file = to_save_filename)

    return(NA)
  })
```




Our tutorial is based on training TabPFN purely on complete cases within an empirical trait dataset. You will first need the following packages.
```
pandas 
math
numpy 
gc
os
tabpfn
```

Imputing species trait data is simple:
- Preprocess your data
  - Compute eigenvectors using 
- Identify complete cases
- Define and fit your TabPFN model
- Use this to impute your full dataset
- Use prior imputations to inform future ones (including in the trait process)
- Save your multiply imputed datasets
- Use each for your analyses and pool using Rubin's' Rules


We then preprocess our dataset, including 




