setwd("C://Users//Lenovo//Desktop//DS//R")
getwd()
install.packages("caret")
library(caret)
install.packages("rpart")
install.packages("rpart.plot")
library(rpart)
install.packages("e1071")
library(e1071)

## Importing Data
data_url <- c("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data")
download.file(url = data_url, destfile = "car.data")
car_df <- read.csv("car.data", sep = ",", header = FALSE)
head(car_df)

#structure of DF
str(car_df)


# All the features are categorical .. no normalization is needed!


## Slicing data to 70:30 Ratio
set.seed(3033)
intrain <- createDataPartition(y=car_df$V7, p=0.7, list = FALSE)
intrain <- createDataPartition(y = car_df$V7, p= 0.7, list = FALSE)
train <- car_df[intrain,]
test <- car_df[-intrain,]

dim(train)
dim(test)



## Preprocessing : Missing values & Attributes with differnt range
anyNA(car_df)


## Dataset Summarized details
summary(car_df)

## Training DT classifier with criterion as information gain
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(3333)
dtree_fit <- train(V7 ~., data = train,method= "rpart",parms=list(split = "information"),trControl = trctrl,tunelength =10)
dtree_fit <- train(V7 ~., data = train, method = "rpart",
                   parms = list(split = "information"),
                   trControl=trctrl,
                   tuneLength = 10)


### Trained Decision Tree classifier results

dtree_fit

## Plot DT using prp() method

install.packages("rpart.plot")
library(rpart.plot)
prp(dtree_fit$finalModel, box.palette = "Reds", tweak = 1.2)

# Prediction
test[1,]

predict(dtree_fit,newdata = test[1,])

# prediction for whole data

test_pred <- predict(dtree_fit, newdata = test)

## For checking Accuracy : Confusion Matrix
confusionMatrix(test_pred, test$V7)



## Training the DT Classifier with Criterion as gini Index
set.seed(3333)

dtree_fit_gini <- train(V7 ~., data = train, method = "rpart",
                        parms = list(split = "gini"),
                        trControl=trctrl,
                        tuneLength = 10)

dtree_fit_gini

## Plot DT

prp(dtree_fit_gini$finalModel, box.palette = "Blues", tweak = 1.2)


## Prediction

test_pred_gini <- predict(dtree_fit_gini, newdata = test)

## Checking Accuracy 
confusionMatrix(test_pred_gini, test$V7)
