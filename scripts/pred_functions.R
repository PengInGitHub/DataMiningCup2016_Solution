##############################################
#                                            #
#  1. FUNCTION FOR CHECKING NEW DATA LEVELS  #
#                                            #
##############################################

# 1. The function compares factor levels, which are present in training and validation data.
# 
# 2. If there are new values in "test" subset, which are not present in "train" subset, the function 
#    returns the list of these levels. The function does not check if there are values, which are present 
#    in training subset only, since that does not prevent prediction algorithms from working.

# introducing the function
check.data <- function(train, test) {
  
  # creating necessary objects
  factors <- rep(FALSE, ncol(train))
  levels  <- array(vector(mode = "list", length = 1), c(1, ncol(train)))
  
  # checking for new factor levels
  for(attr in colnames(train)) {
    if (is.factor(train[[attr]])) {
      
      # comparing levels
      index <- unique(test[[attr]]) %in% unique(train[[attr]])
      new.levels <- unique(test[[attr]])[!index]
      
      # displaying information
      if (length(new.levels) != 0 & length(new.levels) < 5) {
        print(c(paste0(attr, " - ", length(new.levels), " new levels:"), paste(as.character(new.levels), collapse = ", ")))
        factors[which(colnames(train) == attr)] <- TRUE
        levels[[1, which(colnames(train) == attr)]] <- new.levels
      }
      if (length(new.levels) != 0 & length(new.levels) >= 5) {
        print(c(paste0(attr, " - ", length(new.levels), " new levels, including:"), paste(as.character(new.levels[1:5]), collapse = ", ")))
        factors[which(colnames(train) == attr)] <- TRUE
        levels[[1, which(colnames(train) == attr)]] <- new.levels
      }
      if (length(new.levels) == 0) {
        print(paste0(attr, " - no new levels"))
      }
    }
  }
  
  # returning variables
  print("Partitioning is complete.")
  return(list(factors = factors, levels = levels))
}


#############################################
#                                           #
#   2. FUNCTION FOR PREPARING PREDICTIONS   #
#                                           #
#############################################

# 1. The function works with numeric predictions from both regression and classifiation models.
#  
# 2. The function preprocesses vector with predictions and accounts for several data anomalies:
#   - predictions are rounded to the nearest integer
#   - negative predictions are set to 0
#   - too large predictions are set to 5
#   - if predicted returnQuantity > quantity, then prediction = quantity
#   - if quantity = 0, then prediction = 0
#   - if productGroup = Gift Card, then prediction = 0
#
# 3. If model is a binary classifier, it is possible to set a cutoff for prediction rounding.
#    In that case, input vector should contain probabilities of "1" (returning the product).

# introducing the function
prepare.prediction <- function(prediction, test.data, cutoff = 0.5) {
  
  # displaying errors
  if (is.vector(prediction)      == FALSE)   {stop("Predctions have to be a vector")}
  if (is.numeric(prediction)     == FALSE)   {stop("Predctions have to be numeric")}
  if (length(prediction) != nrow(test.data)) {stop("Predctions and data are not the same length")}
  
  # rounding predictions
  if (cutoff == 0.5) {
    prediction <- round(prediction, digits = 0)
  }
  else {
    prediction[prediction >= cutoff] <- 1
    prediction[prediction <  cutoff] <- 0
  }
  
  # eliminating extreme values
  prediction[prediction < 0] <- 0
  prediction[prediction > 5] <- 5
  
  # checking if returnQuantity > quantity
  prediction[prediction > test.data$quantity] <- test.data$quantity[prediction > test.data$quantity]
  
  # imputing zeros for giftcards
  prediction[test.data$productGroup == "GC"] <- 0
  
  # imputing zeros for quantity = 0
  prediction[test.data$quantity == 0] <- 0 
  
  # returning predictions
  return(prediction)
}



#############################################
#                                           #
#     3. FUNCTION FOR ERROR EVALUATION      #
#                                           #
#############################################

# 1. The function returns two error measures: total.error, which is the error function used in competition,
#    and mean.error, which is total.error divided by the number of observations in the testing data.
#
# 2. If predictions are not integers, they are automatically rounded to the nearest integer, 
#    since only natural numbers and zeros are allowed as predictions in the submission file.

# introducing the function
prediction.error <- function(prediction, test.data) {
  
  # displaying errors
  if (is.vector(prediction)      == FALSE)    {stop("Predctions have to be a vector")}
  if (is.numeric(prediction)     == FALSE)    {stop("Predctions have to be numeric")}
  if (length(prediction) != nrow(test.data))  {stop("Predctions and data are not the same length")}
  
  # rounding prediction to the integers
  if (identical(prediction, round(prediction)) == FALSE) {warning("Predictions are not integers. The values are automatically rounded.")}
  prediction <- round(prediction, digits = 0)
  
  # calculating errors
  actuals     <- test.data$returnQuantity
  total.error <- sum(abs(prediction - actuals))
  mean.error  <- total.error/nrow(test.data)
  
  # returning errors
  return(list(total.error = total.error, mean.error = mean.error))
}



#############################################
#                                           #
#   4. FUNCTION FOR EXPORTING PREDICTIONS   #
#                                           #
#############################################

# 1. The function untrims relevant ID variables since they have to be present in submission 
#    dataset in their initial format. If we make additional changes to these variables, they
#    have to be accounted for in the function.
#
# 2. If predictions are not integers, they are automatically rounded to the nearest integer, 
#    since only natural numbers and zeros are allowed as predictions in the submission file.

# introducing the function
save.prediction <- function(prediction, test.data, file.name = "humboldt") {
  
  # displaying error messages
  if (is.vector(prediction)    == FALSE)     {stop("Predctions have to be a vector")}
  if (is.numeric(prediction)   == FALSE)     {stop("Predctions have to be numeric")}
  if (length(prediction) != nrow(test.data)) {stop("Predictions and dataset are not the same length")}
  
  # creadting dataset with relevant coloumns
  test.data <- test.data[, c("orderID", "articleID", "colorCode", "sizeCode")]
  
  # untrimming ID variables
  test.data <- within(test.data, {
    orderID    <- as.character(paste0("a", as.character(orderID)))
    articleID  <- as.character(paste0("i100", as.character(articleID)))
  })
  
  # rounding predictions
  if (identical(prediction, round(prediction)) == FALSE) {warning("Predictions are not integers. The values are automatically rounded.")}
  prediction <- round(prediction, digits = 0)
  
  # adding predictions to the data
  test.data$prediction <- prediction
  
  # exporting predictions
  file.name <- paste(file.name, ".txt", sep = "")
  write.table(test.data, row.names = FALSE, col.names = TRUE, quote = FALSE, sep = ";", file = file.name)
}


#############################################
#                                           #
#    5. FUNCTION TO IMPUTE RETURN RATES     #
#                                           #
#############################################

# introducing the function
add_returns <- function(train, test, standard_value = 0.5) {
  
  # transforming data
  train <- data.table(train)
  test  <- data.table(test)
  
  # calculating return rates
  train$return_per_color <- return_per_color(train)
  train$return_per_size <- return_per_size(train)
  train$return_per_productGroup <- return_per_productGroup(train)
  train$return_per_voucherID <- return_per_voucherID(train)
  train$return_per_paymentMethod <- return_per_paymentMethod(train)
  train$return_per_customerID <- return_per_customerID(train)
  
  # writing to test subcet
  test <- merge(test,unique(train[,.(return_per_color),by=colorCode]), by="colorCode",all.x = TRUE)
  test <- merge(test,unique(train[,.(return_per_size),by=sizeCode]), by="sizeCode",all.x = TRUE)
  test <- merge(test,unique(train[,.(return_per_productGroup),by=productGroup]), by="productGroup",all.x = TRUE)
  test <- merge(test,unique(train[,.(return_per_voucherID),by=voucherID]), by="voucherID",all.x = TRUE)
  test <- merge(test,unique(train[,.(return_per_paymentMethod),by=paymentMethod]), by="paymentMethod",all.x = TRUE)
  test <- merge(test,unique(train[,.(return_per_customerID),by=customerID]), by="customerID",all.x = TRUE)
  
  # imputing missing values
  test[,return_per_color := replace(return_per_color,is.na(return_per_color),standard_value)]
  test[,return_per_size := replace(return_per_size,is.na(return_per_size),standard_value)]
  test[,return_per_productGroup := replace(return_per_productGroup,is.na(return_per_productGroup),standard_value)]
  test[,return_per_voucherID := replace(return_per_voucherID,is.na(return_per_voucherID),standard_value)]
  test[,return_per_paymentMethod := replace(return_per_paymentMethod,is.na(return_per_paymentMethod),standard_value)]
  test[,return_per_customerID := replace(return_per_customerID,is.na(return_per_customerID),standard_value)]
  
  ## imputation for customerID: average return of new customer
  returning_customers <- duplicated(test$customerID)
  newCustomer_return <- test[!returning_customers, mean(returnQuantity, na.rm=TRUE)]
  test[,return_per_customerID := replace(return_per_customerID,is.na(return_per_customerID),newCustomer_return)]
  
  # transforming data
  train <- as.data.frame(train)
  test  <- as.data.frame(test)
  
  # saving data
  return(list(train = train, test = test))
}


#############################################
#                                           #
#  6. SPECIFIC FUNCTIONS FOR RETURN RATES   #
#                                           #
#############################################

# rerurn per voucherID
return_per_voucherID <- function(dt){
  dt[, new := mean(returnQuantity,na.rm=TRUE), by = voucherID]
  dt[is.na(dt$new)]$new <- 0.5
  return(dt$new)
}

# rerurn per size
return_per_size <- function(dt){
  dt[, new := mean(returnQuantity,na.rm=TRUE), by = sizeCode]
  dt[is.na(dt$new)]$new <- 0.5
  return(dt$new)
}

# return per productGroup
return_per_productGroup <- function(dt){
  dt[, new := mean(returnQuantity,na.rm=TRUE), by = productGroup]
  dt[is.na(dt$new)]$new <- 0.5
  return(dt$new)
}

# return per customerID
return_per_customerID <- function(dt){
  dt[, new := mean(returnQuantity,na.rm=TRUE), by = customerID]
  dt[is.na(dt$new)]$new <- 0.5
  return(dt$new)
}

# return per paymentMethod
return_per_paymentMethod <- function(dt){
  dt[, new := mean(returnQuantity,na.rm=TRUE), by = paymentMethod]
  dt[is.na(dt$new)]$new <- 0.5
  return(dt$new)
}

# return per color
return_per_color <- function(dt){
  dt[, new := mean(returnQuantity,na.rm=TRUE), by = colorCode]
  dt[is.na(dt$new)]$new <- 0.5
  return(dt$new)
}


#############################################
#                                           #
#    7. FUNCTIONS FOR ENSEMBLE SELECTION    #
#                                           #
#############################################

# There are two functions: 
# 1) ES performs simple Ensemble Selection.
# 2) BES performs Bagged Ensemble Selection.

# loading library
library(compiler)

# function for perfroming ES
ES <- cmpfun(function(X, Y, iter = 100L){
  
  # setting initial values
  N           <- ncol(X)
  weights     <- rep(0L, N)
  pred        <- 0 * X
  sum.weights <- 0L
  
  # performing hill-climbing
  while(sum.weights < iter) {
    sum.weights   <- sum.weights + 1L
    pred          <- (pred + X) * (1L / sum.weights)
    errors        <- sqrt(colSums((pred - Y) ^ 2L))
    best          <- which.min(errors)
    weights[best] <- weights[best] + 1L
    pred          <- pred[, best] * sum.weights
  }
  
  # returning model weights
  return(weights / sum.weights)
})


# function for performing bagged ES
BES <- cmpfun(function(X, Y, bags = 10L, p = 0.5, iter = 100L, display = TRUE){
  
  # setting initial values
  i <- 0L
  N <- nrow(X)
  M <- ncol(X)
  W <- matrix(rbinom(bags * M, 1, p), ncol = M)
  
  # performing bagging
  while(i < bags)  {
    
    # displyaing iteration number  
    if ((display == TRUE) & ((i+1)%%10L == 0L)) {
      print(paste0("BES - iteration ", i+1, "/", bags))
    }
    
    # doing ES on a bagged sample
    i         <- i + 1L
    ind       <- which(W[i, ] == 1)
    W[i, ind] <- W[i, ind] * ES(X[, ind], Y, iter)
  }
  
  # returning model weights
  return(colSums(W) / bags)
})


#############################################
#                                           #
#      8. FUNCTION FOR PARALLELIZATION      #
#                                           #
#############################################

# function for combining results of parallel processes into lists
comb <- function(x, ...) {
  lapply(seq_along(x), function(i) c(x[[i]], lapply(list(...), function(y) y[[i]])))
}


#############################################
#                                           #
#      9. FUNCTION FOR DATA PARTITIONING    #
#                                           #
#############################################

#Function:
selectPart<- function(tr, part, colNum, cat){
  dataset<- tr[tr$unique_ID %in% part$unique_ID[part[,colNum]==cat],]
  return(dataset)
}