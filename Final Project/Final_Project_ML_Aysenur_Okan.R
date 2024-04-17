---
  title: "Final Project"
author: "Aysenur Okan"
date: "2024-04-16"
output:
  word_document: default
html_document: default
---
  
``` {r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)

library(rpart)
library(rpart.plot)
library(partykit)
library(caret)
library(e1071)
library(party)
library(randomForest)
library(readr)
library(tidyverse)


setwd("/Users/aysenur/Documents/GitHub/Machine_Learning/Final Project")
df <- read_csv("/Users/aysenur/Documents/GitHub/Machine_Learning/Final Project/ml_df.csv")

df$PDWEMA <- factor(df$PDWEMA) #Passive SI
df$SIEMA <- factor(df$SIEMA) #Active SI 


#Split data into training and testing
set.seed(150) # Set a seed for reproducibility
index <- createDataPartition(df$SIEMA, p = 0.5, list = FALSE) # Create an index for splitting the data; use 50% for training and 50% for testing


# Create training and testing datasets
df <- df[index, ] #training data
tsdf <- df[-index, ] #testing data

```


# Fit decision tree using rpart

Biggest predictors of switching in the next trial are value difference in the chosen and the best option, and prediction error. To see if the model overfits the data, I looked at the complexity parameter, which showed me the value of 0.013. The plot shows that there is barely any tuning needed. The prediction errors are very small and the only value under the line is 0.013.  

```{r}
set.seed(41)
tree.rpart=rpart(SIEMA~. -id,data=df) #predict switching choice from all predictors 
prp(tree.rpart) #plot the decision tree
plotcp(tree.rpart)
#(1 s.e. from minimum error in the left out fold) 
printcp(tree.rpart) #similar, but now with numbers, not the plot

```



## Prune tree

To exercise pruning, I use the complexity parameter that was determined via k-fold cross-validation above, an re-plot the tree. All the values are the same as above. 

```{r}
tree.rpartp=prune(tree.rpart,cp=.012) #extract the tree with the determined complexity parameter; #adding "p" to signal that it is the pruned one
prp(tree.rpartp) #plot the pruned tree
```


## Examine the prediction error of the full tree with the test dataset

Next, I examine the prediction error of the full tree and the pruned tree in the testing dataset. 

```{r}
#prediction error for full tree 
pred.rpart=predict(tree.rpart,newdata=tsdf,type='class') #predicted values 
table(pred.rpart,tsdf$SIEMA) #cross-tabulation table (observed values are in columns)
miss.rpart=1-mean(pred.rpart==tsdf$SIEMA, na.rm = T) #0-1 loss 
miss.rpart # misclassification rate
```

## Examine the prediction error the pruned tree in the testing dataset

These are the same because my full tree barely gave me a complexity parameter. Even though I used it to examine whether there would be a difference in prediction error, it was likely since the pruned tree is barely adjusted at all. 

```{r}
#prediction error for pruned tree
pred.rpartp=predict(tree.rpartp,newdata=tsdf,type='class') 
table(pred.rpartp,tsdf$SIEMA)
miss.rpartp=1-mean(pred.rpartp==tsdf$SIEMA, na.rm = T)
miss.rpartp
```




# Bagging with `randomForest()`

```{r}
#bag1=randomForest(SIEMA~.-id,data=df,mtry=7,ntree=500) #bagging model; `mtry` function argument controls the number of predictors considered every time the tree makes a split. If the `mtry` values equals the number of predictors, then the randomForest function does bagging.

#Error above due to missing value; removing and fitting the model again
dfn <- na.omit(df)
tsdfn <- na.omit(tsdf)
bag1 <- randomForest(SIEMA ~ ., data = dfn, mtry = 7, ntree = 500)

bagpred=predict(bag1,tsdfn,type='response') #obtain predicted values from testing dataset 
bag.table=table(bagpred,tsdfn$SIEMA) #cross-tabulation (rows are predicted, columns are observed 
bag.table

1-(mean(bagpred==tsdfn$SIEMA))
varImpPlot(bag1) #plot variable importance measures
```



## Random Forest

```{r}
# Predicting SIEMA with random forest on the training dataset
rf1=randomForest(SIEMA~.-id,data=dfn,mtry=3,ntree=400) #use 3 vars at a time (3 random predictors as candidates to split on), 400 trees grown in bootstrapped datasets. 
#obtain the predicted values in the testing dataset
rfpred=predict(rf1,tsdfn,type='response')
# cross-tabulation of the classes
rf.table=table(rfpred,tsdfn$SIEMA)
#the variable importance measures.
rf.table
1-(mean(rfpred==tsdfn$SIEMA))
varImpPlot(rf1)
```


## The random forest algorithm with conditional inference from cforest()

```{r}
dfn[] <- lapply(dfn, function(x) if(is.character(x)) factor(x) else x)
tsdfn[] <- lapply(tsdfn, function(x) if(is.character(x)) factor(x) else x)

crf1 <- cforest(SIEMA ~ . - id, data = dfn, controls = cforest_unbiased(ntree = 400, mtry = 3))
# Make predictions on the testing dataset
crfpred <- predict(crf1, newdata = tsdfn, type = "response")
crf.table=table(crfpred,tsdfn$SIEMA)
crf.table
1-(mean(crfpred==tsdfn$SIEMA))
vi=varimp(crf1)
dotchart(sort(vi,decreasing=FALSE))
```


##Tuning the tree with caret

```{r}
trctrl=trainControl(method='cv',number=5) #k-fold cross-validation to determine the tuning parameters.
```



#G#row a random forest with randomForest() to predict SIEMA, and try 5 tuning parameters and the previously-specified cross-validation routine.
```{r}
set.seed(2000)
#Fit Model 
train.rf=train(SIEMA~.,data=dfn,method='rf',tuneLength=5,trControl=trctrl) 
#See tuning parameters
train.rf
plot(train.rf)

final.rf=train.rf$finalModel
varImpPlot(final.rf)
```


```{r}
#Fit model for conditional inference random forest
set.seed(2000)
train.crf=train(SIEMA~.,data=dfn,method='cforest',tuneLength=5,trControl=trctrl)
train.crf
plot(train.crf)
final.crf=train.crf$finalModel

#plot variable importance
vi=varImp(final.crf)[,1]
names(vi)=rownames(varImp(final.crf))
dotchart(sort(vi,decreasing=FALSE))
```



