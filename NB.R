library(e1071)
path <- 'https://raw.githubusercontent.com/thomaspernet/data_csv_r/master/data/titanic_csv.csv'
Titanic_df <- read.csv(path)
head(Titanic_df)
str(Titanic_df)
shuffle_index <- sample(1:nrow(Titanic_df))
head(shuffle_index)

Titanic <- Titanic_df[shuffle_index,]
head(Titanic_df)

## Removing home.dest,cabin,name,x,ticket as they are not important variables
library(dplyr)
titanic <- Titanic_df %>%
  select(-c(home.dest,cabin,name,X,ticket)) %>%
  #converting to factor level
  mutate(pclass=factor(pclass,levels = c(1,2,3),labels = c('Upper','Middle','Lower')),
         survived = factor(survived,levels=c(0,1),labels = c('Died','Survived'))) %>%
  na.omit()

## Implementing Naive Bayes model
Naive <- naiveBayes(survived ~., data = titanic)
Naive

## Prediction on the Dataset
Pred <- predict(Naive,titanic)
head(Pred)

## Confusion Matrix
Confusion <- table(Pred,titanic$survived)

##Accuracy Test 77.77
accuracy <- sum(diag(Confusion))/sum(Confusion)



## Naive Bayes using MLR Package
install.packages("mlr")
library(mlr)

# Create a classification task for learning on TItanic
task = makeClassifTask(data = titanic,target = "survived")
#Initialize the Naive Bayes Classifier
selected_model <- makeLearner("classif.naiveBayes")
#Train the Model
NB <- train(selected_model, task)
#Learner model summary
NB$learner.model

#Prediction
Pred_mlr <- as.data.frame(predict(NB,newdata = titanic))

head(Pred_mlr)

# Confusion Matrix
Conf <- table(Pred_mlr[,1],titanic$survived)

conf <- confusionMatrix(Pred_mlr[,1],titanic$survived)

#Accuracy Test = 1 (Perfect Fit)
Accuracy1 <-  sum(diag(Conf))/sum(Conf)
