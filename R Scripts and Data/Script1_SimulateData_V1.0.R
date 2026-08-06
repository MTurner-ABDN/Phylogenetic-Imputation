#Title: Script1_SimulateDataV1.0.R
#Version: 1.0
#Dependencies: load required packages
#Author: Thomas F. Johnson
#Date: 19/02/2019

if (sp == "Mammals"){
  Empirical_mammal <- data.frame(read.table(file = 'Full_mammal_empirical.txt', header = TRUE, sep = '\t', stringsAsFactors = TRUE))
  covariance_data <- cor(Empirical_mammal)
} else if (sp == "Plants"){
  Empirical_plant <- data.frame(read.table(file = 'Full_plant_empirical.txt', header = TRUE, sep = '\t', stringsAsFactors = TRUE))
  covariance_data <- cor(Empirical_plant, use="pairwise.complete.obs")  
  print(covariance_data)
  } 
ntraits <- ncol(covariance_data)

#Simulate traits through a phylogenetic signal
Tree = pbtree(n=num)
Data <- simtraits(ntraits = ntraits, tree = Tree, nreps = 1)
ListOfSpecies = Data$trait_data$species
Traits = data.matrix(Data$trait_data[,-1])

#Standardise correlation between traits
VarianceMatrix = solve(chol(var(Traits)))
UpdateTraits =  Traits %*% VarianceMatrix
TargetCor <- covariance_data #  I've changed this to an empirical correlation
TargetCor <- chol(TargetCor)
AdjustedTraits <- UpdateTraits %*% TargetCor * sd(Traits[,1]) + mean(Traits[,1])
apply(AdjustedTraits, 2, sd)
cor(AdjustedTraits)
rm(TargetCor, VarianceMatrix, UpdateTraits)


#Scale traits between 0 and 10
RangeScale <- function(x){((x-min(x))/(max(x)-min(x)))}
if (sp == "Mammals"){
  ScaledTraits = cbind(RangeScale(AdjustedTraits[,1]),RangeScale(AdjustedTraits[,2]),RangeScale(AdjustedTraits[,3]),RangeScale(AdjustedTraits[,4])
                       ,RangeScale(AdjustedTraits[,5]),RangeScale(AdjustedTraits[,6]))
  } else {
  ScaledTraits = cbind(RangeScale(AdjustedTraits[,1]),RangeScale(AdjustedTraits[,2]),RangeScale(AdjustedTraits[,3]),RangeScale(AdjustedTraits[,4])
                       ,RangeScale(AdjustedTraits[,5]),RangeScale(AdjustedTraits[,6]),RangeScale(AdjustedTraits[,7]))
  }

#Function to specify the response-trait relationship. Code found here https://stats.stackexchange.com/questions/15011/generate-a-random-variable-with-a-defined-correlation-to-an-existing-variable/15040#15040
CorFunction = function(CorRho, ExistingVar, NewVarMean, NewVarSD){
VariableLength = length(ExistingVar)
theta <- acos(CorRho) # corresponding angle
x1 = ExistingVar # fixed given data
x2 = rnorm(VariableLength, NewVarMean, NewVarSD) # new random data
X = cbind(x1, x2) # matrix
Xctr = scale(X, center=TRUE, scale=FALSE) # centered columns (mean 0)
Id = diag(VariableLength) # identity matrix
Q = qr.Q(qr(Xctr[ , 1, drop=FALSE])) # QR-decomposition, just matrix Q
P = tcrossprod(Q) # = Q Q'   # projection onto space defined by x1
x2o = (Id-P) %*% Xctr[ , 2] # x2ctr made orthogonal to x1ctr
Xc2 = cbind(Xctr[ , 1], x2o) # bind to matrix
Y = Xc2 %*% diag(1/sqrt(colSums(Xc2^2))) # scale columns to length 1
FinalVector <- Y[ , 2] + (1 / tan(theta)) * Y[ , 1]  # final new vector
cor(x1, FinalVector) 
return(FinalVector)}

#Specify whether the trait-response relationship should equal 0 or 0.75
if(d == "NoTrend"){
  TrendNumeric = 0
} else {
  TrendNumeric = 0.75
}

#Create Response-trait dataset
if (sp == "Mammals"){
  Data = cbind(ListOfSpecies, as.data.frame(cbind(RangeScale(CorFunction(TrendNumeric, Traits[,1], 5, 1)), ScaledTraits[,1:6])))
  } else {
  Data = cbind(ListOfSpecies, as.data.frame(cbind(RangeScale(CorFunction(TrendNumeric, Traits[,1], 5, 1)), ScaledTraits[,1:7])))
  }

#Identify the nodes within the dataset
CollapseTree = function(Tree){
  tree = Tree
  H<-nodeHeights(tree)
  MaxHeight<-max(H)*0.2
  h1<-which(H[,1]<MaxHeight)
  h2<-which(H[,2]>MaxHeight)
  ii<-intersect(h1,h2)
  nodes<-tree$edge[ii,2]
  getDescendants<-phytools:::getDescendants
  tips<-lapply(nodes,getDescendants,tree=tree)
  tips<-tree$tip.label[sapply(tips,function(x,y) x[x<=Ntip(y)][1],y=tree)]
  tree<-drop.tip(tree,setdiff(tree$tip.label,tips))
  return(tree)
}

CollapsedTree = CollapseTree(Tree)  # Collapse the tree
#plot.phylo(Tree)
plot.phylo(CollapsedTree)

CollapseTips = (CollapsedTree$tip.label)  # Get the labels from the collapsed tree

# Getting the tips from the collapsed tree - where they start in data, w
# here they end, and the number of species in this tip.
TipN = NULL
TipNs = NULL

for (A in CollapseTips){
  StartTipN = which(Data$ListOfSpecies == A)
  TipN = as.data.frame(A)
  TipN = cbind(TipN, StartTipN)
  TipNs = rbind(TipNs, TipN)
} 

shift <- function(x, n){
  c(x[-(seq(n))], rep(NA, n))
}

TipNs$EndTipN = shift(TipNs$StartTipN - 1,1)
TipNs[1,2] = 1
TipNs[length(TipNs$StartTipN),3] = num
TipNs$TipDif = TipNs$EndTipN - TipNs$StartTipN + 1
LoopLength = seq(1,length(TipNs$A),1)

# Getting nodes set up for assigning species to respective nips in the collapsed tree.
AssignedNodes = NULL
for (B in LoopLength){
  NodeNumber = TipNs[B,1]
  Repeats = TipNs[B,4]
  AssignNode = as.data.frame(rep(NodeNumber, Repeats))
  AssignedNodes = rbind(AssignedNodes, AssignNode)
}

#Name columns
colnames(Data)[1] = "TreeTips" 
colnames(AssignedNodes) <- c("Node")

#Create tree order to be able to return traits to order the tree
TreeOrder = seq(1,num,1)
BaseData = cbind(AssignedNodes, Data, TreeOrder)
BaseData$Trend = d

#Create trait breaks to sample big medium and small species proportionally
BaseData$TraitBreaks = ifelse(Data$V2 < 0.333, "Small", ifelse(Data$V2 > 0.667, "Large", "Medium"))

#Remove rownames
rownames(BaseData) <- c()

#Create Data list to keep everything together
BaseDataOutput = summary(lm(V1 ~ V2, data = BaseData))
BaseDataOutput = (BaseDataOutput$coefficients)[2,]
BaseDataOutput = cbind(a,b,c,d,t(BaseDataOutput))
StartingTrend = BaseDataOutput
colnames(StartingTrend) = c("Seed", "CorrelationLevel", "ResponsePresent", "Trend", "Estimate", "SE", "T", "P")
CombinedStartingTrend = rbind(CombinedStartingTrend, StartingTrend)

Data = list(BaseData, Tree, CollapsedTree, StartingTrend)
saveRDS(Data, file = paste("../Data/TemporaryData/",sp,"/Simulation_",sp,"_Seed_",a,"_",num,".rds", sep = ""))


#Only bring BaseData, Tree, CollapsedTree and Starting Trend out of script
rm(CollapseTips, CollapseTree, AssignedNodes, TipNs, TipN, AssignNode, Traits, A, B, ListOfSpecies, LoopLength, NodeNumber, Repeats, StartTipN, shift, RangeScale, CorFunction, TreeOrder, ScaledTraits, AdjustedTraits, BaseDataOutput, TrendNumeric, Data)
