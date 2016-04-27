library(caret)
library(rattle)
library(rpart.plot)

## Loading and reading data

if (! file.exists("training.csv")) {
      url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv" 
      download.file(url,"training.csv")
}

trainingFull <- read.csv("training.csv")
trainingFull <- trainingFull[,-(1:7)]
inTrain <- createDataPartition(trainingFull$classe, p = 0.6, list = FALSE)
training <- trainingFull[inTrain,]
validate <- trainingFull[-inTrain,]

## First prepocessing

## Number of classe variable
## n <- length(names(training))
## classe <- training$classe
## classe_levels <- levels(training$classe)

### Remove NAs
### persentage of NAs in columns
colnas <- apply(training, 2, function(x) mean(is.na(x)) <= .5)
### persentage of columns having more 50% NAs
mean(colnas > .5)
### 
## colnas <- colnas <= .5 ## colnas <- colnas > .5
## table(colnas)
training <- training[,colnas]
validate <- validate[,colnas]
dim(training)
dim(validate)

## Remove nsv

nsv <- nearZeroVar(training,saveMetrics=TRUE)
head(nsv)
training <- training[,!nsv$nzv]
validate <- validate[,!nsv$nzv]
dim(training)
dim(validate)


############# Tree prediction ###########################

rpartModel <- train(classe ~ .,method="rpart", data=training)
rpartModel

fancyRpartPlot(rpartModel$finalModel)

predictTrain1 <- predict(rpartModel,newdata = training)
confusionMatrix(predictTrain1, training$classe)

predictValidate1 <- predict(rpartModel,newdata = validate)
confusionMatrix(predictValidate1, validate$classe)

####################### LDA prediction ######################

ldaModel <- train(classe ~ .,method="lda2", data=training)
ldaModel

predictTrain2 <- predict(ldaModel,newdata = training)
confusionMatrix(predictTrain2, training$classe)

predictValidate2 <- predict(ldaModel,newdata = validate)
confusionMatrix(predictValidate2, validate$classe)

######################## PCA prediction + LDA ##################

n <- length(names(training))
preProc <- preProcess(training[,-n],method=c("pca"),pcaComp=7)
trainingPC <- predict(preProc,training[,-n])
classe <- training$classe 
trainingPC <- cbind(trainingPC, classe)
validatePC <- predict(preProc,validate[,-n])
classe <- validate$classe 
validatePC <- cbind(validatePC, classe)

pcaModel <- train(classe ~ .,method="lda", data=trainingPC)
pcaModel

predictTrain3 <- predict(pcaModel,newdata = trainingPC)
confusionMatrix(predictTrain3, training$classe)

predictValidate3 <- predict(pcaModel,newdata = validatePC)
confusionMatrix(predictValidate3, validate$classe)

##################### Cross-validation ####################### 99,7%

set.seed(753)

# We choose a cross-validation of 4-folds instead of bootstrap default method for
# gradient boosting model (gbm) in order to speed up training.

fitControl <- trainControl(method = "cv", number = 4)

# We provide additionnal parameters to gbm, not far from default ones

gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9),
                        n.trees = (1:15)*100,
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

## system.time(
cvModel <- train(classe ~. ,data = training ,method="gbm" ,trControl = fitControl ,verbose = FALSE,tuneGrid = gbmGrid)


predictTrain4 <- predict(cvModel,newdata = training)
confusionMatrix(predictTrain4, training$classe)

predictValidate4 <- predict(cvModel,newdata = validate)
confusionMatrix(predictValidate4, validate$classe)

################ Random forest ####################### 99,3%

## library(doMC)
## registerDoMC(cores = 4)

rfModel <- train(classe ~ .,method="rf", data=training)
rfModel

predictTrain5 <- predict(rfModel,newdata = training)
confusionMatrix(predictTrain5, training$classe)

predictValidate5 <- predict(rfModel,newdata = validate)
confusionMatrix(predictValidate5, validate$classe)

############## Load and prepare testing dataset ##############################

if (! file.exists("plm-testing.csv")) {
      url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv" 
      download.file(url,"plm-testing.csv")
}

testing <- read.csv("plm-testing.csv")
testing <- testing[,-(1:7)]
testing <- testing[,colnas]
testing <- testing[,!nsv$nzv]

dim(testing)

## Prediction for testing dataset 
predictTest6 <- predict(rfModel,newdata = testing)
confusionMatrix(predictTest6, testing$classe)
predictTest6

predictTest7 <- predict(cvModel,newdata = testing)
confusionMatrix(predictTest7, testing$classe)
predictTest7

