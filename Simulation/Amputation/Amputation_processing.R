# Loading in the simulated rds file - extracting just the trait data:

groups <- list("Plants", "Mammals")
Seeds <- list(1,2,3,4,5,6,7,8,9,10)
species <- list(200, 1000)

# Get the simualted traits:

for (group in groups){
  for (seed in Seeds){
    for (num in species){
      curr_file <- paste0("../Simulation/Data/",group,"/Simulation_",group,"_Seed_",seed,"_",num,".rds", sep ="")
      print(curr_file)
      curr_data <- readRDS(curr_file)
      if (group == "Mammals"){
        curr_traits <- curr_data[[1]][, c(4,5,6,7,8,9)] # Extract traits
      } else {
        curr_traits <- curr_data[[1]][, c(4,5,6,7,8,9,10)]
      }
      to_save_directory <- paste0("../Simulation/Data/",group,"/Simulation_",group,"_Seed_",seed,"_",num,".txt", sep ="")
      write.table(curr_traits,  to_save_directory, sep="\t", col.names = TRUE)
    }
  }
}

# Update .rds with amputed traits

scenario <- c("MAR", "MNAR", "Mix")
missingness <- c(5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80)

for (group in groups){
  for (seed in Seeds){
    for (num in species){
      for (sc in scenario) {
        for (ms in missingness) {
          # Current rds
          curr_file <- paste0("../Simulation/Data/",group,"/Simulation_",group,"_Seed_",seed,"_",num,".rds", sep ="")
          curr_data <- readRDS(curr_file)
          curr_amputed_loc <- paste0(group,"/",sc,"/Missing_",ms,"/Amputed_",group,"_Seed_",seed,"_",num,"_",sc,"_missing_",ms,".txt", sep="")
          curr_amputed <- read.table(curr_amputed_loc, sep="\t")
          if (group == "Mammals"){
            curr_data[[1]][, c(4,5,6,7,8,9)] <- curr_amputed
          } else {
            curr_data[[1]][, c(4,5,6,7,8,9,10)] <- curr_amputed
          }
          to_save_directory <- paste0("../Data/Amputed_rds/",group,"/",sc,"/Missing_",ms,"/Simulated_",group,"_Seed_",seed,"_",num,"_",sc,"_Missing_",ms,".rds", sep ="")
          saveRDS(curr_data,  to_save_directory)
        }
      }
    }
  }
}

