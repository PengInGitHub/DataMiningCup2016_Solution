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
library(LiblineaR)
library(data.table)
library(Matrix)
library(SparseM)
library(foreach)
library(doParallel) 


# registering the cores as workers
coreCount <- detectCores(logical = FALSE)
registerDoParallel(coreCount)

# choose how many ways to split the trees 
# !!keep in mind that there must be a clean division of ntree metaparameters!!
# marginal performance gains may diminish if the process count exceeds the core count
treesplit <- 2

# setting directory
setwd("/Users/TJM/Dropbox/R/04_prediction_methods/infrastructure_parallelization/work_in_progress")

# loading functions
source("pred_0_functions.R")
source("pred_0_return_functions.R")

# loading data
load("data_2_5_full.Rda")

# splitting training and testing
class <- df[ is.na(df$returnQuantity),]
train <- df[!is.na(df$returnQuantity),]
train <- train[1:10000,]
rm(df)

# listing variables which are not used in models
unused.vars <- c("unique_ID", "returnQuantity", "orderID", "orderDate", "articleID", "voucherID", 
                 "customerID", "colorCode", "special_group")

# model parameters
# add additional metaparameter values to test iterations!
f.trees  <- seq(200, 500, 50)
f.mtry   <- c(3)

# other parameters
data.ratio <- 0.6
trials     <- 1


#############################################
#                                           #
#         2. CREATING SOME OBJECTS          #
#                                           #
#############################################

# creating parameter grid
f.grid <- expand.grid(mtry = f.mtry, ntree = f.trees)

# calculating number of models
models <- nrow(f.grid)

# matrix with error distributions
trial.errors <- matrix(nrow = trials, ncol = models)

# setting up the clock
time.set <- array(vector(mode = "list", length = 1), c(trials, 10))


#############################################
#                                           #
#           3. MODELLING TRIALS             #
#                                           #
#############################################

# starting the loop
for (t in 1:trials) {
  
  # diplaying the trial number
  print(paste0("STARTING TRIAL ", t, " OUT OF ", trials))
  time.set[[t, 1]] <- proc.time()
  
  
  #############################################
  #                                           #
  #          3.1. DATA PARTITIONING           #
  #                                           #
  #############################################
  
  # marking unused variables
  drops <- names(train) %in% unused.vars
  
  # partitioning
  t.factors <- TRUE
  while (length(t.factors) != 0) {
    
    # partitioning the data
    data.part <- createDataPartition(train$returnQuantity, p = data.ratio, list = F)
    t.valid   <- train[-data.part,]
    t.train   <- train[ data.part,]
    
    # checking unknown categories
    unknowns  <- check.data(t.train[!drops], t.valid[!drops])
    t.factors <- colnames(t.valid[!drops])[unknowns$factors == TRUE]
  }
  
  # diplaying information
  print(paste0("Preparing the data..."))
  
  # creating features and imputing values
  data    <- add_returns(t.train, t.valid)
  t.train <- data$train
  t.valid <- data$test
  t.train <- t.train[, !colnames(t.train) %in% "new"]
  rm(data)
  
  # sorting coloumns in datasets
  t.train   <- t.train[,   order(names(t.train))]
  t.valid   <- t.valid[,   order(names(t.valid))]
  
  # remarking unused variables
  drops <- names(t.train) %in% unused.vars
  
  # removing some objects
  rm(unknowns, t.factors, data.part)
  
  
  #############################################
  #                                           #
  #           3.2. CREATING OBJECTS           #
  #                                           #
  #############################################
  
  # creating error matrix
  errors <- vector(mode = "numeric", length = models)
  names(errors) <- rep("NA", length(errors))
  names(errors)[1:nrow(f.grid)] <- "RF"
  
  # creating predictions matrix
  predictions <- matrix(nrow = nrow(t.valid),   ncol = models)
  colnames(predictions) <- names(errors) 
  
  # information
  time.set[[t, 2]] <- proc.time()
  time.trial <- proc.time() - time.set[[t, 1]]
  print(paste0("Data preparation took ", round(time.trial[3]/60, digits = 0), " minutes."))
  #beep(5)
  
  
  #############################################
  #                                           #
  #  3.3. BASE MODELS: TRAINING & PREDICTING  #
  #                                           #
  #############################################
  
  
  ####### RF parallel iteration loop
  print("Training RF with parallel forest growth...")
  
  # outer loop iterating across rf metaparameters
  # outer loop left at sequential (%do%) by default to reduce overhead and memory strain
  # can be configured as a parallel nested loop by changeing to %dopar%, but watch the process count
  # treesplit*nrow(f.grid) to know how many processes will activate when outer loop is %dopar%
  tmp_rf <- foreach(i = 1:nrow(f.grid), .packages = "randomForest",
                    .export = c("treesplit","f.grid", "t.train", "t.valid")) %do% {
                      
                      print(paste0("Model ", i, " out of ", nrow(f.grid)))
                      
                      # inner loop splitting the rf model
                      m.forest2 <- foreach(ntree=rep(f.grid$ntree[i]/treesplit, treesplit), .combine=combine,
                                           .packages="randomForest",
                                           .export = c("f.grid", "t.train", "t.valid")) %dopar% {
                                             
                                             # training random forest
                                             randomForest(returnBin ~ ., data = t.train[!drops], ntree = ntree, mtry = f.grid$mtry[i])
                                             
                                           }
                      f.predictions <- predict(m.forest2, newdata = t.valid,   type = "prob")[,"1"]
                      return(f.predictions)
                      
                    }
  
  # saving RF predictions
  k <- 0
  for (i in 1:nrow(f.grid)) {
    predictions[, k+i] <- (tmp_rf[[i]])
  }
  
  
  # information
  time.set[[t, 3]] <- proc.time()
  time.trial <- proc.time() - time.set[[t, 2]]
  print(paste0("RF took ", round(time.trial[3]/60, digits = 0), " minutes."))
  
  # clearing the memory
  rm(tmp_rf)
  
  
  # Ensembling techniques were removed for now
  
  
  #############################################
  #                                           #
  #   3.4. ANALYSING PREDICTIVE PERFORMANCE   #
  #                                           #
  #############################################
  
  
  # preparing predictions
  predictions <- apply(predictions, 2, function(x) prepare.prediction(x, test.data = t.valid))
  
  # saving error measures
  errors <- apply(predictions, 2, function(x) prediction.error(x, test.data = t.valid)$total.error)
  
  # diagram with error distribution
  par(mfrow = c(1,1), oma = c(0, 0, 0, 0), xpd = 0)
  plot(errors, type = "p", cex = 2, col = "black", ylab = "Total Error", xlab = "", xaxt = "n", pch = 20)
  #abline(v = c((2+nrow(f.grid)-0.5), (length(errors)-4.5), (length(errors)-1.5)))
  abline(h = min(errors), col = "red", lty = 2)
  axis(1, at = 1:length(errors), labels = names(errors), las = 2)
  
  # memorizing error values
  colnames(trial.errors) <- names(errors)
  trial.errors[t,] <- errors
  
  # stopping the clock
  time.end <- proc.time()
  time.trial <- time.end - time.set[[t, 1]]
  sprintf("Total time to run trial %.0f: %.0f minutes (%.1f hours).", t, time.trial[3]/60, time.trial[3]/3600)
  
  # displaying the best method
  sprintf(paste0("The best algorithm is ", names(which.min(errors)), " with error = ", min(errors)))
  #beep(3)
  
  # ending the loop
}


#############################################
#                                           #
#      4. VISUALISATION AND COMAPRISON      #
#                                           #
#############################################

# boxplots with all models
par(mfrow = c(1,1), oma = c(0, 0, 0, 0), xpd = 0)
boxplot(trial.errors, use.cols = TRUE, col = "grey", names = colnames(trial.errors), xaxt = "n", main = "Error Distributions", horizontal = FALSE, ylab = "Total Error", xlab = "")
#the division lines could be designed to scale with the batches based on ntree, but do not do so yet
#abline(v = c((2+nrow(f.grid)-0.5), (length(errors)-4.5), (length(errors)-1.5)))
axis(1, at = 1:length(errors), labels = names(errors), las = 2)


# stopping the clock
time.final <- proc.time()
time.all <- time.final - time.start
sprintf("Total time to run the code: %.0f minutes (%.1f hours).", time.all[3]/60, time.all[3]/3600)
#beep(8)

# saving relevant objects
save(trial.errors, file = "rf_ntree_errors.Rda")
