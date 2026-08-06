library("ape")
library("dispRity")
library("stats")
library("phytools")
library("Rphylopars")
library("castor")
library("ggplot2")
library("ellipse")
library("missMDA")
library("TreeTools")
library("data.table")
library("matrixcalc")
library("Matrix")
library("geiger")
library("rlist")
library("ratematrix")
library("mvMORPH")
library("stringr")

check.evo.model <- function(Tree, traits, trait_name){
  Tree <- curr_Tree
  traits <- curr_Traits
  trait_name <- colnames(curr_Traits[trait_name])
  curr_trait_vector <- setNames(traits[[trait_name]],traits[, "species"])
  curr_trait_vector <- curr_trait_vector[!is.na(curr_trait_vector)]
  
  species_with_data <- names(curr_trait_vector)
  pruned_tree <- drop.tip(Tree, setdiff(Tree$tip.label, species_with_data))
  models <- c("BM", "OU")
  
  fitBM <- fitContinuous(pruned_tree, curr_trait_vector, model="BM")
  fitOU <- fitContinuous(pruned_tree, curr_trait_vector, model="OU")

  aics <- matrix(c(fitBM$opt$aic, fitOU$opt$aic)))
  best_model <- models[which(aics==min(aics))]
  
  return(best_model)
}

add.internal.node <- function(tree, element.IDs, split.length) {
  
  # Find the edge row
  old_edge <- which(
    tree$edge[,1] == element.IDs[1] &
      tree$edge[,2] == element.IDs[2]
  )
  
  if(length(old_edge) == 0)
    stop("edge not found")
  
  # Original length
  L <- tree$edge.length[old_edge]
  
  if(split.length >= L)
    stop("split length must be smaller than original edge length")
  
  # Define new node number
  new_node <- max(tree$edge) + 1
  
  # Remove old edge + length
  tree$edge <- tree$edge[-old_edge, , drop = FALSE]
  tree$edge.length <- tree$edge.length[-old_edge]
  
  # Add two new edges
  new_edges <- rbind(
    c(element.IDs[1], new_node),
    c(new_node, element.IDs[2])
  )
  
  new_lengths <- c(split.length, L - split.length)
  
  # Append to tree
  tree$edge <- rbind(tree$edge, new_edges)
  tree$edge.length <- c(tree$edge.length, new_lengths)
  
  tree$Nnode <- tree$Nnode + 1
  
  return(tree)
}

pclust <- function(point,km,df){
  get.p <- function(clust,x){
    d         <- as.numeric(x-km$centers[clust,])
    sigma.inv <- solve(cov(df[km$cluster==clust,]))
    X.sq      <- d %*% sigma.inv %*% d
    p         <- pchisq(X.sq,length(d),lower.tail=FALSE)
  }
  sapply(1:max(km$cluster),get.p,x=point)
}

validate_VS_nodes <- function(new_species, tip_num, N_traits, curr_tip, species_num){
  
  new_species <- data.frame(new_species)
  
  ground_truth <- N_traits[(species_num+1):nrow(N_traits), ]
  colnames(new_species) <- colnames(ground_truth)
  
  pca <- prcomp(ground_truth, scale. = T, retx=T)  # Do a PCA of the ground truth data
  summary(pca)$importance[3,1:4] # Summary stats of the PCA - which PC's are most important - 93% of variation is explained in first two 
  
  scores <- data.frame(pca$x)[1:2]  # So use the firwst two components
  km <- kmeans(scores, centers=1, nstart = 25)  # Do kmeans clustering on the principle components of the ground truth data
  
  pc.test <- predict(pca, new_species)[,1:2]  # Make predicitons using test_data
  rownames(pc.test) <- c(LETTERS[1:2])
  
  p <- apply(pc.test, 1, pclust, km=km, df=scores)
  return(data.frame(p))
}

compare.correlations <- function(original_data, expanded_data, B = 1000){
  original_data <- original_trait_data[,2:(num_traits+1)]
  expanded_data <- expanded_traits[,1:num_traits]
  p <- ncol(original_data)
  diff_array <- array(NA, dim = c(p, p, B))
  
  # Number of new rows
  n_orig <- nrow(original_data)
  n_exp <- nrow(expanded_data)
  n_new <- n_exp - n_orig
  
  new_rows <- expanded_data[(n_orig+1):n_exp, ]
  b <- 1
  for (b in 1:B){
    
    boot_orig <- original_data[sample(n_orig, replace = TRUE), ]
    boot_new <- new_rows[sample(n_new, replace = TRUE), ]
    boot_combined <- rbind(boot_orig, boot_new)
    
    cor_combined <- cor(boot_combined, use="pairwise.complete.obs")
    diff_array[ , , b] <- cor_combined - cor(boot_orig, use="pairwise.complete.obs")
  }
  
  lower_diff <- apply(diff_array, c(1,2), quantile, .025)
  upper_diff <- apply(diff_array, c(1,2), quantile, .975)
  
  sig_change <- (lower_diff > 0) | (upper_diff < 0)

  return(any(sig_change))  # TRUE if any pairwise correlation changed significantly
}


modified.simulate.species <- function(Num_to_sim, phylo_tree, trait_data, species_num, num_traits, cor_struc, evo_process, models){

  # Starting variables
  new_species_IDs <- list()
  min_tree <- read.tree(text = "(A:0.1,B:0.1);")
  
  Tree <- phylo_tree
  TraitData <- trait_data
  original_trait_data <- TraitData[c(2:(num_traits+1))]
  original_tip_names <- TraitData$species
  
  batch_size <- 20
  accepted_sims <- 0
  attempts <- 0
  max_attempts <- 200
  max_node_attempts <- 200
  max_species_attempts <- 200
  
  my_correlations <- cor_struc
  TraitData[,-1] <- lapply(TraitData[,-1], function(x) as.numeric(as.character(x)))

  Fit <- phylopars(trait_data = TraitData, tree = Tree, model = "BM")  
  Node_traits <- Fit$anc_recon
  
  #my_correlations <- Fit$phylocov
  original_Nodes <- Node_traits
  
  while (accepted_sims < Num_to_sim){
    
    attempts <- attempts + 1
    if (attempts > max_attempts){
      print("Maximum batch attempts reached")
      print(paste0("Successfully simulated: ",accepted_sims," species.", sep =""))
      final_tree <- Tree
      final_trait_data <- Node_traits[1:(Num_to_sim+nrow(trait_data)),]
      new_species_names <- unlist(new_species_IDs)
      return(list(final_tree, final_trait_data, new_species_names))
    }
    
    cat("Starting new batch\n")
    
    batch_tree <- Tree
    batch_Node_traits <- Node_traits
    batch_species_IDs <- list()
    batch_species_IDs <- list()
    
    i <- 1
    batch_failed <- FALSE
    
    while (i <= batch_size) {
      
      species_retry_attempts <- 0
      max_species_retries <- 200
      success <- FALSE
      
      while (!success) {
        
        species_retry_attempts <- species_retry_attempts + 1
        
        if (species_retry_attempts > max_species_retries) {
          cat("Too many failures for this species - restarting batch\n")
          batch_failed <- TRUE
          break
        }
        
        nSim <- accepted_sims + i
        
        curr_tip <- sample(batch_tree$tip.label, 1)
        tip_num <- which(batch_tree$tip.label == curr_tip)
        parent_node <- batch_tree$edge[batch_tree$edge[,2] == tip_num, 1]
        
        old_edge <- which(batch_tree$edge[,1] == parent_node & batch_tree$edge[,2] == tip_num)
        curr_max_edge_length <- batch_tree$edge.length[old_edge]
        
        ntips <- length(batch_tree$tip.label)
        tip_parent_lengths <- batch_tree$edge.length[batch_tree$edge[,2] <= ntips]
        
        length_from_parent_node <- runif(1, min(tip_parent_lengths)*0.9, curr_max_edge_length*0.99)
        
        updated_tree <- add.internal.node(batch_tree,
                                          element.IDs = c(parent_node, tip_num),
                                          split.length = length_from_parent_node)
        
        min_tree_node <- read.tree(text = paste("(A:",length_from_parent_node,",B:",length_from_parent_node,");"))
        
        parent_trait_data <- unlist(batch_Node_traits[parent_node,])
        
        sim_node <- TRUE
        node_attempts <- 0
        max_node_attempts <- 200
        
        while (sim_node) {
          
          node_attempts <- node_attempts + 1
          
          if (node_attempts > max_node_attempts) {
            cat("Node simulation failed - retrying species\n")
            break
          }
          
          simulated_traits <- mvSIM(min_tree_node, nsim = 1, model = models,
                                    param = list(theta = parent_trait_data,
                                                 sigma = my_correlations))
          
          validation_results_n <- validate_VS_nodes(
            data.frame(simulated_traits),
            tip_num,
            original_Nodes,
            curr_tip,
            species_num)
          
          if (validation_results_n[1,1] > 0.01 && validation_results_n[2,1] > 0.01){
            simulated_node <- simulated_traits[sample(c(1,2),1),]
            sim_node <- FALSE
            
          } else if (validation_results_n[1,1] >= 0.01){
            simulated_node <- simulated_traits[1,]
            sim_node <- FALSE
            
          } else if (validation_results_n[2,1] >= 0.01){
            simulated_node <- simulated_traits[2,]
            sim_node <- FALSE
          }
        }
        
        if (sim_node) next   # retry same species
        
        New_node_ID <- parent_node + 1
        
        new_tip_number <- (max(as.numeric(gsub("[^0-9.]", "", batch_tree$tip.label))) + 1)
        new_tip_label <- paste0("t", new_tip_number)
        
        new_tip_edge_length <- runif(1, 0.3, 0.7)
        
        new_updated_tree <- AddTip(updated_tree,
                                   label = new_tip_label,
                                   where = New_node_ID,
                                   edgeLength = new_tip_edge_length)
        
        min_tree_tip <- read.tree(text = paste("(A:",new_tip_edge_length,",B:",new_tip_edge_length,");"))
        
        
        sim_species <- TRUE
        species_attempts <- 0
        max_species_attempts <- 200
        
        while (sim_species) {
          
          species_attempts <- species_attempts + 1
          
          if (species_attempts > max_species_attempts) {
            cat("Tip simulation failed - retrying species\n")
            break
          }
          
          simulated_tip_traits <- mvSIM(min_tree_tip, nsim = 1, model = models,
                                        param = list(theta = simulated_node,
                                                     sigma = my_correlations))
          
          validation_results_n_tips <- validate_VS_nodes(
            data.frame(simulated_tip_traits),
            tip_num,
            original_Nodes,
            curr_tip,
            species_num)
          
          if (validation_results_n_tips[1,1] > 0.01 && validation_results_n_tips[2,1] > 0.01){
            simulated_tip <- simulated_tip_traits[sample(c(1,2),1),]
            sim_species <- FALSE
            
          } else if (validation_results_n_tips[1,1] >= 0.01){
            simulated_tip <- simulated_tip_traits[1,]
            sim_species <- FALSE
            
          } else if (validation_results_n_tips[2,1] >= 0.01){
            simulated_tip <- simulated_tip_traits[2,]
            sim_species <- FALSE
          }
        }
        
        if (sim_species) next   # retry same species
        
        
        batch_species_IDs[[i]] <- new_tip_label
        
        batch_tree <- collapse.singles(new_updated_tree)
        
        curr_tip_data <- transpose(as.data.frame(simulated_tip))
        rownames(curr_tip_data) <- new_tip_label
        colnames(curr_tip_data) <- colnames(batch_Node_traits)
        
        batch_Node_traits <- rbind(curr_tip_data, batch_Node_traits)
        
        rn <- rownames(batch_Node_traits)
        is_num <- grepl("^\\d+$", rn)
        rn[is_num] <- as.character(as.numeric(rn[is_num]) + 1)
        rownames(batch_Node_traits) <- rn
        
        curr_trait_data <- as.data.frame(t(simulated_node))
        rownames(curr_trait_data) <- (nrow(batch_Node_traits)+1)
        colnames(curr_trait_data) <- colnames(curr_tip_data)
        
        
        batch_Node_traits <- rbind(batch_Node_traits, curr_trait_data)
        
        success <- TRUE
      }
      
      if (batch_failed) break
      
      i <- i + 1
    }
    
    expanded_traits <- batch_Node_traits[1:(nrow(original_trait_data)+accepted_sims+batch_size), ]
    sig_change <- modified.compare.correlations(original_Nodes, rbind(expanded_traits[1:(accepted_sims+batch_size),1:num_traits], original_Nodes), original_tip_names)

    if (sig_change == FALSE){
      
      cat("Batch accepted\n")
      
      Tree <- batch_tree
      Node_traits <- batch_Node_traits
      
      new_species_IDs <- c(new_species_IDs, batch_species_IDs)
      accepted_sims <- accepted_sims + batch_size
      
      print("Successfully updated correlation structure is: ")
      print(cor(Node_traits))
      
    } else {
      cat("")
    }
  }
  
  final_tree <- Tree
  final_trait_data_sim <- Node_traits[1:(Num_to_sim),]
  final_trait_data_full <- Node_traits[1:(Num_to_sim+nrow(trait_data)),]
  new_species_names <- unlist(new_species_IDs)
  
  return(list(final_tree, final_trait_data_full, final_trait_data_sim, new_species_names))
}


modified.compare.correlations <- function(original_data, expanded_data, original_tip_names, tip_weight = 10, node_weight = 1, B = 500, tol = 0.1) {
  
  p <- ncol(original_data)
  n_orig <- nrow(original_data)
  n_exp <- nrow(expanded_data)
  
  # Identify tips
  orig_is_tip <- rownames(original_data) %in% original_tip_names
  
  # Weights
  orig_weights <- ifelse(orig_is_tip, tip_weight, node_weight)
  orig_weights <- orig_weights / sum(orig_weights)
  
  # Bootstrap arrays
  cor_boot_orig_array <- array(NA, dim = c(p, p, B))
  cor_boot_exp_array  <- array(NA, dim = c(p, p, B))
  
  for (b in 1:B) {
    # Original bootstrap
    boot_orig <- original_data[sample(1:n_orig, n_orig, replace = TRUE, prob = orig_weights), ]
    cor_boot_orig_array[,,b] <- cor(boot_orig, use = "pairwise.complete.obs")
    
    # Expanded bootstrap
    boot_exp <- expanded_data[sample(1:n_exp, n_exp, replace = TRUE), ]
    cor_boot_exp_array[,,b] <- cor(boot_exp, use = "pairwise.complete.obs")
  }
  
  # Compare means 
  orig_mean <- apply(cor_boot_orig_array, c(1,2), mean)
  exp_mean  <- apply(cor_boot_exp_array,  c(1,2), mean)
  
  observed_diff <- mean(abs(orig_mean - exp_mean))

  diff_null <- replicate(B, {  # Natural variability
    boot1 <- original_data[sample(1:n_orig, n_orig, replace = TRUE, prob = orig_weights), ]
    boot2 <- original_data[sample(1:n_orig, n_orig, replace = TRUE, prob = orig_weights), ]
  
    mean(abs(cor(boot1) - cor(boot2)))
  })

  threshold <- quantile(diff_null, 0.95)  # Significant difference in distributions?
  sig_diff <- observed_diff > threshold
  
  return(sig_diff)
}


for (g in c("Mammals", "Plants")){
  if (g == "Mammals"){
    n <- 7
  } else {
    n <- 8
  }
  
  for (num in c(200, 1000)){  
    for (seed in c(1,2,3,4,5,6,7,8,9,10)){ 
      for (sc in c("MAR", "MNAR", "Mix")){ 
        for (ms in c(5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80)){ 
          print(paste0("Working on ",g," with ",num," species, seed ",seed,". Missing conditions are ",sc," at ",ms,"%", sep=""))
          num_to_sim <- min(c(num/2, 300))
          curr_directory <- paste0("Data/Amputed/",g,"/",sc,"/Missing_",ms,"/Simulated_",g,"_Seed_",seed,"_",num,"_",sc,"_Missing_",ms,".rds", sep="")
          curr_RDS <- readRDS(curr_directory)
          
          curr_Tree <- curr_RDS[[2]]
          all_nan_found <- FALSE
          
          if (g == "Mammals"){
            curr_Traits <- curr_RDS[[1]][, c(2,4,5,6,7,8,9)]
            colnames(curr_Traits) <- c("species", "V2", "V3", "V4", "V5", "V6", "V7")
          } else {
            curr_Traits <- curr_RDS[[1]][, c(2,4,5,6,7,8,9,10)]
            colnames(curr_Traits) <- c("species", "V2", "V3", "V4", "V5", "V6", "V7", "V8")
          }
          
          
          for (col in c(2:n)){
            if (sum(is.na(curr_Traits[,col])) == num){
              all_nan_found <- TRUE
              print("All nan column found, skipping.")
            }
          }
          
          if (all_nan_found) next
          
          rownames(curr_Traits) <- curr_Traits[[1]]
          temp_traits <- curr_Traits[, c(2:n)]

          model_vector <- character(n - 1)
          
          for (i in 2:n) {
            model_vector[i - 1] <- tryCatch({
              curr_model <- check.evo.model(curr_Tree, curr_Traits, i)
              if (curr_model != "EB") {
                paste0(curr_model, "1")
              } else {
                curr_model
              }
            }, error = function(cond) {
              message("Error: ", cond$message)
              "BM1"
            })
          }
          
          # Get evolutionary correlation matrix
          traits <- as.matrix(temp_traits)
          p <- ncol(temp_traits)
          Sigma <<- matrix(NA, p, p)  # assign globally
          
          for(i in 1:p){
            for(j in i:p){
              if(i == j){
                keep <- complete.cases(traits[,i])
                if(sum(keep) < 2){next}
                sub_traits <- traits[keep, i, drop = FALSE]
                sub_tree <- drop.tip(curr_Tree, setdiff(curr_Tree$tip.label, rownames(sub_traits)))
                fit <- mvMORPH::mvBM(sub_tree, sub_traits, model="BM1")
                Sigma[i,i] <- fit$sigma[1,1]
              } else {
                keep <- complete.cases(traits[,c(i,j)])
                if(sum(keep) < 2){next}
                sub_traits <- traits[keep, c(i,j)]
                sub_tree <- drop.tip(curr_Tree, setdiff(curr_Tree$tip.label, rownames(sub_traits)))
                contrasts_i <- pic(sub_traits[,1], sub_tree)
                contrasts_j <- pic(sub_traits[,2], sub_tree)
                Sigma[i,j] <- cov(contrasts_i, contrasts_j)
                Sigma[j,i] <- Sigma[i,j]
              }
            }
          }
          
          # Fill remaining NA values
          diag_mean <- mean(Sigma[!is.na(Sigma)], na.rm = TRUE)
          for(i in 1:p){
            if(is.na(Sigma[i,i])){
              Sigma[i,i] <- diag_mean * 0.5
            }
          }
          
          for(i in 1:p){
            for(j in i:p){
              if(is.na(Sigma[i,j])){
                if(i == j){
                  print("ignore diag")
                } else {
                  Sigma[i,j] <- sqrt(Sigma[i,i]*Sigma[j,j]) * 0.5
                  Sigma[j,i] <- Sigma[i,j]
                }
              }
            }
          }
          #print(Sigma)
          to_use_sigma <- Sigma
          
          
          is_PD <- is.positive.definite(to_use_sigma)
          if (is_PD == "FALSE"){
            PD_correlations <- nearPD(to_use_sigma)
            to_use_sigma <- as.matrix(PD_correlations$mat)
          }
          
          curr_Traits$species <- as.numeric(str_sub(curr_Traits$species, 2, -1))
          curr_Tree$tip.label <- curr_Traits$species
          
          str(curr_Traits)
          
          tryCatch({
            outs <- modified.simulate.species(num_to_sim, curr_Tree, curr_Traits, num, num_traits = (ncol(curr_Traits)-1), cor_struc = to_use_sigma, models = model_vector)
            output_dir <- paste0("Simulated/",g,"/",sc,"/Missing_",ms,"/Simulated_",g,"_Seed_",seed,"_",num,"_",sc,"_Missing_",ms,"_with_VS.rds", sep="")
            simulated_traits <- outs[[3]]
            expanded_with_missing <- rbind(simulated_traits, curr_Traits[, c(2:n)])
            outs[[2]] <- expanded_with_missing
            print("Did we get here?")
            saveRDS(outs, output_dir)
          },error = function(cond){
            print("Error occurred")
            print(cond)
          })

        }
      }
    }
  }
}
