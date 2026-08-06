library("MPSEM")

groups <- c("Mammals", "Plants")
species <- c(200,1000)
seeds <- c(1,2,3,4,5,6,7,8,9,10)
scenarios <- c("MAR", "MNAR","Mix")
missingness <- c(5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80)

normalize <- function(x){
  return((x - min(x))/ (max(x) - min(x)))
}


for (g in groups) {
  
  if (g == "Mammals"){
    traits <- c("V2","V3","V4","V5","V6","V7")
  } else if (g == "Plants") {
    traits <- c("V2","V3","V4","V5","V6","V7","V8")    
  }
  
  for (num in species) {
    for (seed in seeds) {
      for (sc in scenarios) {
        for (ms in missingness){
          for (trait in traits) {
            # Load the current eigenvector data
            curr_eigens_location <- paste0("Data/TemporaryData/",g,"/Eigens/",sc,"/Missing_",ms,"/Trimmed_",g,"_Eigens_Seed_",seed,"_",num,"_",sc,"_missing_",ms,"_trait_",trait,".rds", sep="")
            curr_eigens <- readRDS(file = curr_eigens_location)
            current_eigen_data <- as.data.frame(curr_eigens$u)
            normalized_eigens <- apply(current_eigen_data, 2, normalize)
            
            # Load corresponding amputed traits
            curr_traits_location <- paste0("Path_to_tree_trait.rds", sep="")
            curr_traits <- readRDS(file = curr_traits_location)

            if (g == "Mammals"){
              curr_traits <- curr_traits[[1]][, c(2,4,5,6,7,8,9)]
            } else if (g == "Plants") {
              curr_traits <- curr_traits[[1]][, c(2,4,5,6,7,8,9,10)]    
            }
            
            # combine
            combined_traits_eigens <- cbind(curr_traits, normalized_eigens)
            
            out_loc <- paste0("Full_eigens/",g,"/",sc,"/Missing_",ms,"/Combined_Eigens_traits_Seed_",seed,"_",num,"_",sc,"_Missing_",ms,"_trait_",trait,".txt", sep="")
            write.table(x = combined_traits_eigens, file = out_loc, row.names = FALSE, col.names = TRUE, sep = "\t")

          }
        }
      }
    }
  }
}