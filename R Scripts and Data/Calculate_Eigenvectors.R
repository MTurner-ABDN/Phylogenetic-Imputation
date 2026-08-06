#Title: Script4_CalculateEigenvectors_V1.0.R
#Version: 1.0
#Dependencies: BaseData, DeletedData, Tree
#Author: Thomas F. Johnson
#Date: 19/02/2019

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


RobustPEM = function(Response, TGraph){
  TrimmedPEM = PEM.fitSimple(
        y = Response,
        x = NULL,
        w = TGraph,
        d = "distance", sp = "species")
  Error_In_La.svd = "No"
  PEM = list(TrimmedPEM, Error_In_La.svd)
  return(PEM)
} 


species_num <- list(200, 1000)
Seeds <- list(1,2,3,4,5,6,7,8,9,10)
scenarios <- list("MAR", "MNAR", "Mix")
missing_rates <- list(5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80)


#for (group in list("Mammals")){
for (group in list("Mammals")){
  if (group == "Mammals"){  
    traits <- c("V2", "V3", "V4", "V5", "V6", "V7")
  } else if (group == "Plants") {
    traits <- c("V2", "V3", "V4", "V5", "V6", "V7", "V8")
  }

  print(paste0("Working on ", group ,sep=""))
  for (num in species_num) {
    print(paste0("With speices number ", num ,sep=""))
    for (seed in Seeds){ 
      print(paste0("Seed ", seed ,sep=""))

      for (scenario in scenarios) {
        print(paste0("amputed under ", scenario ,sep=""))
        for (missingness in missing_rates) {
          print(paste0("With missingness ", missingness ,sep=""))

          print("Working on next pass: start")

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
          }
        }
      }
    }
  }
}

