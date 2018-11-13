---
title: "Practical Machine Learning Course Project"
output: 
  html_document: 
    keep_md: yes
---
#Practical Machine Learning Course Project

##Overview
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. Subjects were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/harIn. The goal of this project will be to develop a model that will accuarately predict if a person falls into one of the 5 categories.  


##Exploratory Analysis


```r
library(caret)
library(randomForest)
library(gbm)
```

Download data

```r
train.data<- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"), header = TRUE)

test.data<- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"), header = TRUE)
```

Cleaned data by removing variables that consisted mainly of NA values, had near zero variance, and non-relevant variables such as names and timestamps.

```r
## removing variables that are mostly NA
var.NA<- sapply(train.data, function(x) mean(is.na(x)))>  0.95

train.data<- train.data[ , var.NA == F]

## removing variables with Near Zero Variance

NZV<- nearZeroVar(train.data)

train.data<- train.data[ , -NZV]

## removing variables that are just for identification or timestamps

train.data<- train.data[, -(1:5)]
```

Partitioned the training data into a train and test subset for machine learning.

```r
set.seed(430)
Partition<- createDataPartition(train.data$classe, p = 0.7, list = FALSE)
training.set<- train.data[Partition, ]
testing.set<- train.data[-Partition, ]
```
## Prediction Models

Used a 3 fold cross-validation to optimize tuning paramters for models.


```r
## Cross validation
set.seed(430)
cross.val<- trainControl(method = "cv", number= 3, verboseIter = FALSE)
```

### Recursive Partitioning Model
Has a poor accuarcy of 49.6%, so it is not a good prediction model.

```r
set.seed(430)

model.fit.rpart<- train(classe ~., data= training.set, method ="rpart", trControl = cross.val)

predict.test.rpart<- predict(model.fit.rpart, newdata= testing.set)

confusionMatrix(testing.set$classe, predict.test.rpart)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1527   25  118    0    4
##          B  476  392  271    0    0
##          C  469   32  525    0    0
##          D  429  184  351    0    0
##          E  166  160  283    0  473
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4957          
##                  95% CI : (0.4828, 0.5085)
##     No Information Rate : 0.5212          
##     P-Value [Acc > NIR] : 1               
##                                           
##                   Kappa : 0.3407          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.4979  0.49433  0.33915       NA  0.99161
## Specificity            0.9478  0.85330  0.88448   0.8362  0.88739
## Pos Pred Value         0.9122  0.34416  0.51170       NA  0.43715
## Neg Pred Value         0.6343  0.91551  0.78946       NA  0.99917
## Prevalence             0.5212  0.13475  0.26304   0.0000  0.08105
## Detection Rate         0.2595  0.06661  0.08921   0.0000  0.08037
## Detection Prevalence   0.2845  0.19354  0.17434   0.1638  0.18386
## Balanced Accuracy      0.7229  0.67381  0.61181       NA  0.93950
```

### Generalized Boosted Model

Has an accuracy of 98.4%, so it could be used as a model, but there is a good likelihood of misidentification.

```r
model.fit.gbm<- train(classe ~., data= training.set, method ="gbm", trControl = cross.val)
predict.test.gbm<- predict(model.fit.gbm, newdata= testing.set)
```

```r
confusionMatrix(testing.set$classe, predict.test.gbm)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671    3    0    0    0
##          B   14 1107   18    0    0
##          C    0   15 1002    8    1
##          D    0    3    7  952    2
##          E    0    7    1    6 1068
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9856          
##                  95% CI : (0.9822, 0.9884)
##     No Information Rate : 0.2863          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9817          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9917   0.9753   0.9747   0.9855   0.9972
## Specificity            0.9993   0.9933   0.9951   0.9976   0.9971
## Pos Pred Value         0.9982   0.9719   0.9766   0.9876   0.9871
## Neg Pred Value         0.9967   0.9941   0.9946   0.9972   0.9994
## Prevalence             0.2863   0.1929   0.1747   0.1641   0.1820
## Detection Rate         0.2839   0.1881   0.1703   0.1618   0.1815
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9955   0.9843   0.9849   0.9915   0.9971
```
## Random Forest Model

Has an accuracy of 99.8%. Because of this accuracy this is the model that will be used to compare to the testing data.

```r
set.seed(430)
model.fit.rf<- train(classe ~., data= training.set, method ="rf", trControl = cross.val)

predict.test.rf<- predict(model.fit.rf, newdata= testing.set)

confusionMatrix(testing.set$classe, predict.test.rf)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    1 1135    3    0    0
##          C    0    4 1022    0    0
##          D    0    0    2  962    0
##          E    0    0    0    3 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9978          
##                  95% CI : (0.9962, 0.9988)
##     No Information Rate : 0.2846          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9972          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9965   0.9951   0.9969   1.0000
## Specificity            1.0000   0.9992   0.9992   0.9996   0.9994
## Pos Pred Value         1.0000   0.9965   0.9961   0.9979   0.9972
## Neg Pred Value         0.9998   0.9992   0.9990   0.9994   1.0000
## Prevalence             0.2846   0.1935   0.1745   0.1640   0.1833
## Detection Rate         0.2845   0.1929   0.1737   0.1635   0.1833
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9997   0.9978   0.9972   0.9982   0.9997
```

## Prediction

We fit our testing data to the random forest prediction model that we used above and found the predicted "classe" for 20 points to be

```r
#Predict Test

predict.test<- predict(model.fit.rf, newdata = test.data)

matrix(predict.test)
```

```
##       [,1]
##  [1,] "B" 
##  [2,] "A" 
##  [3,] "B" 
##  [4,] "A" 
##  [5,] "A" 
##  [6,] "E" 
##  [7,] "D" 
##  [8,] "B" 
##  [9,] "A" 
## [10,] "A" 
## [11,] "B" 
## [12,] "C" 
## [13,] "B" 
## [14,] "A" 
## [15,] "E" 
## [16,] "E" 
## [17,] "A" 
## [18,] "B" 
## [19,] "B" 
## [20,] "B"
```

