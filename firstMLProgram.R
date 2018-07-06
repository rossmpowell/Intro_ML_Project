#oading ML package caret, and loading the iris dataset into the script
library(kernlab)
library(e1071)
library(caret)
library(plyr)
library(dplyr)
library(recipes)
data(iris)
dataset <- iris

#Splitting the dataset into the training data and the test data

validation_index <- createDataPartition(dataset$Species, p=0.80, list=FALSE)
training_data <- dataset[validation_index,]
validation_data <- dataset[-validation_index,]

#Let's check out some of the attributes of this dataset

#First gives us the x by y dimensions of the table
#Next shows us the different class that each column is in the dataset
#Shows us the first few rows of the dataset
#Shows us the unique values of the dataset
#Next two lines show create a table of values or the percentage of each Species in the dataset
#Finally we see the summary statistics for the dataset

dim(dataset)
sapply(dataset, class)
head(dataset)
levels(dataset$Species)
percentage <- prop.table(table(dataset$Species)) * 100
cbind(freq=table(dataset$Species), percentage=percentage)
summary(dataset)

#####################################---------------------------------------------------

#Visualization part of this. Let's see if we can confirm some of the results from above
#and see if we can start to group some of that data together based on class

#First lets divide the dataset into the x variables and the y variable.

# The measurements of the different flowers will be our x variables
# The resulting class will be the y variable

x <- training_data[,1:4]
y <- training_data[,5]

# Remember, our goal is to create a system that will take in the training data in order to 
# create a model that we can use to predict the class of the validation data!

#par is the R layout function, this is saying that the graphics we need are going to be laid out
#in a 1,4 matrix where the plots will be created by the following for function, which will graph a box
#plot by each x variable. It can do this because we just took all of the x variables out and put them into 
#the x matix defined above

par(mfrow=c(1,4))
   for(i in 1:4) {
   boxplot(x[,i], main = names(iris)[i])
}

#Now lets create a different plot
plot(y)

#This plot is designed to check the authenticity of the breakdown of the different types of the Species
# As we can see, we were correct in our summary statistics

#Now time for a more advanced plot. This will group the different flowers into ovals in order to see some
#trends in the different x attributes. This is the scatterplot matrix!
featurePlot(x=x, y=y, plot="ellipse")

#This graph really shows the difference in the species of Iris. It looks like we will
#be able to create a good model for predicting Iris flowers!

#Here is another one which shows the box plots
featurePlot(x=x, y=y, plot="box")

#Final plot is the density plot, first line creates the scales, second one
#creates the plot
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

##############################-----------------------------------------

#Time for the ML stuff! To start we need to setup a test harness
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

#Now we evaluate, we will test 5 different methods used in predictive analysis
#LDA - Linear Discriminant Analysis
#CART - Classification and Regression Trees
#kNN - k-Nearest Neighbors
#SVM - Support Vector Machines with a linear kernal
#RF - Random Forest

#We will build them using the same seed, which is a way of splitting up the data
#by setting the seed to the same thing each time we test a model

#First is the linear algorithm

set.seed(7)
fit.lda <- train(Species~., data=training_data, method="lda", metric=metric, trControl=control)

#Second are the non-linear algorithms

set.seed(7)
fit.cart <- train(Species~., data=training_data, method="rpart", metric=metric, trControl=control)
set.seed(7)
fit.knn <- train(Species~., data=training_data, method="knn", metric=metric, trControl=control)

#Third are the advanced algorithms
#set.seed(7)
#fit.svm <- train(Species~., data=training_data, method="svmRadical", metric=metric, trControl=control)
set.seed(7)
fit.rf <- train(Species~., data=training_data, method="rf", metric=metric, trControl=control)

#We now can look at the results from these tests to get an idea of which model fits the data best
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, rf=fit.rf))
summary(results)

#Graphic to illustrate the differences, clearly lda is the best model 
dotplots(results)

#Print out some of the results from this
print(fit.lda)

#Let's see how well this model holds up against the other 20% of the data!
predictions(fit.lda, validation_data)
confusionMatrix(predictions, validation$Species)



