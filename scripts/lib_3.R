#############################################
#                                           #
#              1. INTRODUCTION              #
#                                           #
#############################################

# clearing the memory
rm(list = ls())
time.start <- proc.time()

# libraries
library(beepr)
library(caret)
library(randomForest)
library(xgboost)
library(data.table)
library(Matrix)
library(SparseM)
library(foreach)
library(doParallel) 

# setting directory
setwd("/Users/Kozodoi/Desktop/Final Models")
#setwd("/home/RDC/kozodoin.hub/Desktop/Final Models")
#setwd("C:/Users/kozodoin.hub/Desktop/Final Models")

# loading functions
source("pred_functions.R")

# loading data
load("data_2_8_full.Rda")

# loading feature sets
load("var_ranks_2_4.Rda")
load("features_2_4_rfopt.Rda")
load("errors_2_4_xgb.R")

# splitting training and testing
d.class <- df[ is.na(df$returnQuantity),]
d.label <- df[!is.na(df$returnQuantity),]
rm(df)

# listing variables which are not used as predictors
unused.vars <- c("unique_ID", "returnQuantity", "orderID", "orderDate", "articleID", "voucherID", 
                 "customerID", "colorCode", "special_group")

# model parameters
f.trees  <- c(100)
f.mtry   <- c(9:11)
x.depth  <- c(10)
x.eta    <- c(0.5)
x.iters  <- c(300)
x.lambda <- c(700, 800, 900)



#############################################
#                                           #
#            2. CREATING  OBJECTS           #
#                                           #
#############################################

# creating parameter grids
f.grid <- expand.grid(mtry = f.mtry, ntree = f.trees)
x.grid <- expand.grid(depth = x.depth, eta = x.eta, iters = x.iters, lambda = x.lambda)
grid   <- list(f.grid, x.grid) 

# calculating number of models
models <- nrow(f.grid) + nrow(x.grid)

# creating model arrays
m.forest <- array(vector(mode = "list", length = 1), c(2, nrow(f.grid)))
m.xgb    <- array(vector(mode = "list", length = 1), c(2, nrow(x.grid)))

# setting up the clock
time.set <- array(vector(mode = "list", length = 1), c(1, 10))
time.set[[1, 1]] <- time.start



#############################################
#                                           #
#             3. DATA PREPARATION           #
#                                           #
#############################################

# diplaying information
print("Preparing the data...")

# marking unused variables
drops <- names(df) %in% unused.vars

# creating features and imputing values [level 1]
data    <- add_returns(d.label, d.class)
d.label <- data$train
d.class <- data$test
d.label <- d.label[, !colnames(d.label) %in% "new"]
rm(data)

# sorting coloumns in datasets
d.label   <- d.label[,   order(names(d.label))]
d.class  <- d.class[, order(names(d.class))]

# remarking unused variables
drops <- names(d.label) %in% unused.vars

# saving feature lists
x.vars <- c(names(var.ranks[1:which.min(x.errors)]), "returnBin", "C1", "C2", "C3", "C4", "C5", "saswoXt")
r.vars <- attr(terms(r.vars), "term.labels")
r.vars <- paste(r.vars, collapse = " + ")
r.vars <- paste("returnBin ~ ", r.vars, " + C1 + C2 + C3 + C4 + C5 + saswoXt", sep="")
r.vars <- as.formula(r.vars)

# additional data transformations [level 1]
d.class$returnBin <- "NA"
x.y <- as.numeric(d.label$returnBin)-1
x.x <- sparse.model.matrix(returnBin ~ . -1, data = d.label[x.vars])
x.v <- sparse.model.matrix(returnBin ~ . -1, data = d.class[x.vars])

# creating error matrix
errors <- vector(mode = "numeric", length = models)
names(errors) <- rep("NA", length(errors))
names(errors)[(1):(nrow(f.grid))] <- "RF"
names(errors)[(1+nrow(f.grid)):(length(errors))] <- "XGB"

# creating predictions matrix
predictions <- matrix(nrow = nrow(d.class),   ncol = models)
colnames(predictions) <- names(errors) 

# information
time.set[[1, 2]] <- proc.time()
time.trial <- proc.time() - time.set[[1, 1]]
print(paste0("Data preparation took ", round(time.trial[3]/60, digits = 0), " minutes."))
#beep(5)



#############################################
#                                           #
#            4.1. BASE MODELS: RF           #
#                                           #
#############################################

# information
print("Training random forest...")
k <- 0

# training models
for (i in 1:nrow(f.grid)) {
  
  # displaying model number
  print(paste0("Model ", i, " out of ", nrow(f.grid)))
  
  # training full models
  m.forest[[2, i]] <- randomForest(r.vars, data = d.label[!drops],   ntree = f.grid$ntree[i], mtry = f.grid$mtry[i])
  
  # extracting predictions
  predictions[, k+i]  <- predict(m.forest[[2, i]], newdata = d.class,   type = "prob")[,"1"]
}

# information
time.set[[1, 3]] <- proc.time()
time.trial <- time.set[[1, 3]]  - time.set[[1, 2]]
print(paste0("RF took ", round(time.trial[3]/60, digits = 0), " minutes."))
#beep(5)



#############################################
#                                           #
#         4.2. BASE MODELS: XGB             #
#                                           #
#############################################

# information
print("Training xgboost...")
k <- 0 + nrow(f.grid)

# training xgboost
for (i in 1:nrow(x.grid)) {
  
  # displaying model number
  print(paste0("Model ", i, " out of ", nrow(x.grid)))
  
  # training models
  m.xgb[[2, i]] <- xgboost(data = x.x,   label = x.y,   nrounds = x.grid$iters[i], lambda = x.grid$lambda[i], eta <- x.grid$eta[i], max.depth <- x.grid$depth[i], objective = "binary:logistic", verbose = 0)
  
  # extracting predictions
  predictions[, k + i] <- predict(m.xgb[[2, i]], newdata = x.v)
}

# information
time.set[[1, 4]] <- proc.time()
time.trial <- time.set[[1, 4]]  - time.set[[1, 3]]
print(paste0("XGBoost took ", round(time.trial[3]/60, digits = 0), " minutes."))
#beep(5)



#############################################
#                                           #
#    6. ANALYSING PREDICTIVE PERFORMANCE    #
#                                           #
#############################################

# preparing predictions
predictions_prep <- apply(predictions, 2, function(x) prepare.prediction(x, test.data = d.class, cutoff = 0.5))

# saving relevant objects
save(predictions,      file = "tmp2_predictions_raw_3.Rda")
save(errors,           file = "tmp2_errors_3.Rda")
save(grid,             file = "tmp2_parameters_3.Rda")
