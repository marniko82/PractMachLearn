### Preliminary actions

setwd("C:/Users/Nikolici/Desktop/PractMachLearn/project")
library(ggplot2); library(caret) 
set.seed(696969)
train <- read.csv("./pml-training.csv")
test <- read.csv("./pml-testing.csv")

### Extraction of wanted variables

train_short <- cbind(train[,which(grepl("_x",names(train)))],
                     train[,which(grepl("_y",names(train))&!grepl("_ya",names(train)))  ],
                     train[,which(grepl("_z",names(train)))],
                     classe=train[,160])


### split into new train and test set
inTrain = createDataPartition(train_short$classe, p=0.80, list=FALSE)
tts = train_short[inTrain,]
oob_test = train_short[-inTrain,]

#inTrain = createDataPartition(train_short$classe, p=0.01, list=FALSE)
#tts <- train_short[inTrain,]
### Model selection 
all_oob <- matrix(data = NA, nrow = 3, ncol = 1, byrow = TRUE)
fitControl <- trainControl(method="repeatedcv", number=5, repeats=1, verboseIter=TRUE)

## Random Forest
start.time <- Sys.time()
set.seed(696969)
modFitForest <- train(classe~., method="rf",data=tts, trControl=fitControl, verbose=FALSE)
predict_oob <- predict(modFitForest,newdata=oob_test)
all_oob[1,1] <- 1-sum(predict_oob==oob_test$classe)/length(predict_oob)
end.time <- Sys.time()
time.taken <- end.time - start.time; time.taken
saveRDS(modFitForest, "Initial_RF.rds")

## GBM 
start.time <- Sys.time()
set.seed(696969)
gbmGrid <-  expand.grid(n.trees = 100, interaction.depth = 1,shrinkage = 0.1)
modFitGBM <- train(classe~., method="gbm",data=tts, trControl=fitControl, verbose=FALSE)
predict_oob <- predict(modFitGBM,newdata=oob_test)
all_oob[2,1] <- 1-sum(predict_oob==oob_test$classe)/length(predict_oob)
end.time <- Sys.time()
time.taken <- end.time - start.time; time.taken
saveRDS(modFitGBM, "Initial_GBM.rds")

## SVM
start.time <- Sys.time()
set.seed(696969)
modFitSVM <- train(classe~., method="svmRadial",data=tts, trControl=fitControl, verbose=FALSE)
predict_oob <- predict(modFitSVM,newdata=oob_test)
all_oob[3,1] <- 1-sum(predict_oob==oob_test$classe)/length(predict_oob)
end.time <- Sys.time()
time.taken <- end.time - start.time; time.taken
saveRDS(modFitSVM, "Initial_SVM.rds")

# collect resamples
results <- resamples(list(RF=modFitForest, GBM=modFitGBM, SVM=modFitSVM))
# summarize the distributions
summary(results)
# boxplots of results
bwplot(results)
# dot plots of results
dotplot(results)

dimnames(all_oob) <- list(c("RF","GBM","SVM"),c("OOB")) ; all_oob <- data.frame(all_oob)
qplot(rownames(all_oob),OOB, data=all_oob, xlab="Method", main="Comparison of OOB error")
write.table(all_oob,file="all_oob.Rda")

### Model fit 2nd time using the entire train set
start.time2 <- Sys.time()
set.seed(696969)
modFitFinal <- train(classe~., method="rf",data=train_short, trControl=fitControl, verbose=FALSE)
end.time2 <- Sys.time()
time.taken2 <- end.time2 - start.time2
time.taken2
varImp(modFitFinal)
modFitFinal$finalModel

### Prepare test set in the same way as train

test <- test[,-160]
test_short <- cbind(test[,which(grepl("_x",names(test)))],
                    test[,which(grepl("_y",names(test))&!grepl("_ya",names(test)))  ],
                    test[,which(grepl("_z",names(test)))])


### Prediction
final_pred <- predict(modFitFinal,newdata=test_short); final_pred
saveRDS(modFitFinal, "Final.rds")

