
#Attach all packages
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


#Create directory. Required space 22gb
dir.create("../Data")
dir.create("../Data/TemporaryData")
dir.create("../Data/RawData")
dir.create("../Data/SummarisedData")
dir.create("../Results")
dir.create("../Results/ResidualPlots")
dir.create("../Results/RidgePlots")
dir.create("../Results/ModelOutputs")
dir.create("../Results/ModelOutputs/ResidualPlots")
dir.create("../Results/SummaryPlots")


#Save session
LoadedPackages = sessionInfo()
saveRDS(LoadedPackages,"../Results/SessionInfo.rds")


#Create files to fill
CombinedStartingTrend = NULL
CombinedDeletionReport = NULL
CombinedEigenReport = NULL
CombinedImputationReport = NULL


#Run scripts
Seeds = list(1,2,3,4,5,6,7,8,9,10)
num_species = list(200, 400, 600, 800, 1000, 5000)
species = list("Mammals", "Plants")
ResponsePresent = list("No")
Trend = list("PositiveTrend")
b = "empirical"
sim = TRUE
c = "No"
d = "PositiveTrend"

for(sp in species){
  message(paste("Current species: ",sp, sep = ""))
  for(num in num_species){
    for(a in Seeds){
      set.seed(a)
      message(paste("Current seed: ",a, sep = ""))
      for(c in ResponsePresent){
        message(paste("  Response present: ",c, sep = ""))
        for(d in Trend){
          message(paste("   Trend type: ",d, sep = ""))
          message(paste("    ",Sys.time()))
        
        #Simulate datasets. Output: BaseData, Tree, CollapsedTree
        print("Step 1: Simulate datasets")
        source(file="Script1_SimulateData_V1.0.R")
    
        }
      }
      
    }
  }
}
 
saveRDS(CombinedStartingTrend, "../Data/RawData/StartingTrend.rds")
saveRDS(CombinedDeletionReport, "../Data/RawData/DeletionReport.rds")
saveRDS(CombinedEigenReport, "../Data/RawData/EigenReport.rds")
saveRDS(CombinedImputationReport, "../Data/RawData/ImputationReport.rds")