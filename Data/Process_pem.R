library(MPSEM)

group <- list("Mammals", "Plants")  
Seeds <- c(1,2,3,4,5,6,7,8,9,10)
num_species <- c(200,1000)
scenarios <- list("MAR", "Mix", "MNAR")
missingness <- c(5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80)



for (g in group){
  for (num in num_species){
    for (seed in Seeds) {
      for (scenario in scenarios) {
        for (missing in missingness){
          if (g == "Mammals") {
            traits <- list("2", "3", "4", "5", "6", "7")
          } else {
            traits <- list("2", "3", "4", "5", "6", "7", "8")
          }
          for (trait in traits) {
            SimulatedPEM <- readRDS(file = paste("../Simulation/Data/",g,"/Eigens/",scenario,"/Missing_",missing,"/Trimmed_",g,"_Eigens_Seed_",seed,"_",num,"_",scenario,"_missing_",missing,"_trait_V",trait,".rds", sep = ""))
            BaseData <- readRDS(file = paste0("../Data/Amputed_rds/",g,"/",scenario,"/Missing_",missing,"/Simulated_",g,"_Seed_",seed,"_",num,"_",scenario,"_Missing_",missing,".rds", sep = ""))
            BaseData <- BaseData[[1]]
            
            if (g == "Mammals") {
              Base_for_check <- BaseData[,c(4,5,6,7,8,9)]
            } else {
              Base_for_check <- BaseData[,c(4,5,6,7,8,9,10)]
            }
            
            
            most_complete <- names(which.min(colSums(is.na(Base_for_check))))  # Find column with fewest NAs
            
            # So, we want the eigenvectors that explain the most variation
            BasePEM_LM <- tryCatch(
              lmforwardsequentialsidak(
                y = BaseData[[paste0("V", trait)]],
                object = SimulatedPEM,
                alpha = 0.01
              ),
              error = function(e1) {
                tryCatch(
                  lmforwardsequentialsidak(
                    y = BaseData[[most_complete]],
                    object = SimulatedPEM,
                    alpha = 0.01
                  ),
                  error = function(msg) {
                    message(paste0("Error when getting lm eigens for group ",g,", species num ",num,", for seed ",seed," and trait ",trait,". Missingness conditions: ",scenario,", missing ",missing,"%.", sep=""))
                  })
              })
            
            lm_Eigens <- as.data.frame(BasePEM_LM$qr$qr)[,-1]  # Get lm output dataframe
            
            eigens_to_use <- colnames(lm_Eigens)  # Eigenvector names after lm
            base_eigens <- as.data.frame(SimulatedPEM)  # Base eigenvector data
            Eigens <- base_eigens[, eigens_to_use]  # Subset columns using best eigenvectors
            
            
            TreeTips <- as.character(rownames(Eigens))
            Data <- BaseData[, c(4,5,6,7,8,9)]  # Get just trait data
            Data <- cbind(TreeTips, Data)  # And tree tips names
            FullData <- cbind(Data, Eigens)  # And reduced number of eigens  all in one dataset
            write.table(FullData, file = paste("../Data/Process_PEM_outs/",g,"/",scenario,"/Missing_",missing,"/Amputed_Eigens_for_",g,"_Seed_",seed,"_",num,"_",scenario,"_missing_",missing,"_trait_V",trait,".txt", sep = ""))
          }
        }
      }
    }
  }
}