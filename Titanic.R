library(tidyverse)
library(stringr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

train <- read.csv('DA_titanic_train.csv')
test <- read.csv('DA_titanic_test.csv')

# creating a Survived column, filling it with NA so we can attach this to the train data
test$Survived <- NA

# combining the two for pre-processing, the test values will be the ones with missing survival, so
# we can easily separate the two after pre-processing
main_data <- rbind(train, test)

# SibSp: # of Siblings and Spouses on ship
# Parch: # of Parents and Children on ship
# Fare: Fare paid
# Embarked: 3 possible cities

# male <- 1, female <- 0
main_data$Sex <- ifelse(main_data$Sex == 'male', 1, 0)

# C <- 0, Q <- 1, S <- 2 
main_data$Embarked <- sapply(as.character(main_data$Embarked), switch, 
                             "C" = 0, "Q" = 1, "S" = 2, USE.NAMES = F)# %>% as.numeric()

main_data$Embarked <- as.numeric(as.character(main_data$Embarked))

# adding an index column to use later
main_data$Index <- c(1:nrow(main_data))

summary(main_data)

# % of people who survived the tragedy
survivalRate <- length(which(main_data[, 2] == 1)) / nrow(main_data)

# we have 130 NA in age, around 400 in cabin, 1 in embarked ~ TRAIN VALUES
sapply(main_data, function(x) sum(is.na(x)))

# we have 47 NA in age, a lot in cabin and 1 in embarked ~ TEST VALUES
sapply(test, function(x) sum(is.na(x)))

# DEALING WITH NA VALUES IN AGES:

# most young boys have "Master." in their name, people with "Miss." in their name have a younger 
# average age than "Mrs." so giving mean values to NA ages could be more accurate with name "types"
# than just giving a global average to everyone whose age is missing

# global mean age is 29.7
globalMean <- mean(main_data$Age, na.rm = TRUE) %>% round(1)

# possible age indicators in names; in regular expressions '.' means 'any', we need to escape by '\\'
possibleNames <- c('Mr\\.', 'Mrs\\.', 'Master\\.', 'Miss\\.', 'Dr\\.', 'Rev\\.', 'Col\\.')

# the average ages of people with each "name type"
avgAges <- sapply(possibleNames, function(i){
  # true-false list, given index is true if person has the current "name type"
  who <- str_detect(main_data$Name, regex(i, ignore_case = FALSE))
  # rounding and returning the average age
  avgAge <- round(mean(main_data$Age[who], na.rm = TRUE), 1)
})

getNameType <- function(name){
  # loop through possible names
  type <- lapply(1:length(possibleNames), function(i){
    # if name type (ex.: Mr.) is found in name return i to keep the index of the name type found
    if(str_detect(name, regex(possibleNames[i], ignore_case = FALSE))){
      return(i)
    }
  })
}

# returns a vector of type-index pairs ex.: (type: 1 index: 12) .. (type: 6 index: 32) in format: 1 12 2 14 4 19
# type can be 1-7 (possibleNames list)
typeIndexPairs <- sapply(1:nrow(main_data), function(i){
  if(is.na(main_data$Age[i])){
    # the type of the name (1-7)
    x <- getNameType(main_data$Name[i])
    # the index the name type is at
    y <- list(x, i)
  }
}) %>% unlist()

# looping through previous vector by 2, if we start from 1 we get types, if we start from 2 we get indexes
getElementsUnlistedNames <- function(i){
  x <- sapply(seq(i, length(typeIndexPairs), 2), function(i){
    typeIndexPairs[i]
  })
  return(x)
}

# dataframe which contains indexes of name types and their average ages
ageDf <- data.frame(Age = avgAges, Type = c(1:length(avgAges)))

# dataframe with the indexes and types in it
types <- getElementsUnlistedNames(1)
indexes <- getElementsUnlistedNames(2)
indexAndType <- data.frame(Type = types, Index = indexes)

# left join with possibleNames on "Type"value, thus getting the indexes of people with NA in age from main_data 
# and the "name type" that person has
indexAndType <- merge(x = indexAndType, y = data.frame(Type = possibleNames), by = "Type", all.x = TRUE)

# second left join with ageDf, now getting the appropriate age for each NA age person
indexAndType <- merge(x = indexAndType, y = ageDf, by = "Type", all.x = TRUE)

# left join with main_data, by the common Index value, now we have 2 columns with alternating age and NA values
main_data <- merge(x = main_data, y = indexAndType, by = "Index", all.x = TRUE)

# by using the coalesce function we can merge the two age columns, the NA value gets thrown away with each row
# and we keep the one which is not NA
main_data <- main_data %>% mutate(Age = coalesce(Age.x,Age.y)) %>% select(-c(Age.x, Age.y, Type, Ticket, Cabin))

# if there are any NA age values left we give them the global mean age
main_data[is.na(main_data$Age), 6] <- globalMean

# through the previous steps we eliminated NA from age and gave the people with no age an estimate according
# to their names

# the most common element in main_data$Embarked
mostCommonEmbarkPont <- as.numeric(names(sort(-table(main_data$Embarked)))[1])

# the one NA value in main_data$Embarked should get the most common point
main_data$Embarked[is.na(main_data$Embarked)] <- mostCommonEmbarkPont

# by looking at this we can conclude the following:
# on average more people died with lower class (2-3) tickets
# way more males have died than females
# people with parents/children on the ship had slightly lower chances of death
# the survivors paid a higher fare than the people who didn't survive -> connected to difference between classes
# people who didn't survive the tragedy had slightly higher average age
survivalAverages <- sapply(colnames(main_data[,c(3,4,6,7,8,9,10,11)]), function(i){
  survived <- mean(main_data[which(main_data$Survived == 1), i], na.rm = TRUE)
  died <- mean(main_data[which(main_data$Survived == 0), i], na.rm = TRUE)
  
  data <- c(survived, died)
}) %>% data.frame()

# heatmap of the correlation between the numeric values
#library(heatmaply)
#heatmaply_cor(cor(main_data[,c(3,4,6,7,8,9,10,11)]))

# we are predicting chance of death, not survival
main_data$Died <- 1-main_data$Survived

# MOST PRE-PROCESSING STOPS HERE ----------------------

# original test values have no survival data, we can not use them to train a model
originalTest <- main_data[which(is.na(main_data$Died)), ]
originalTrain <- main_data[which(!is.na(main_data$Died)), ]

# removing pre-processing data
rm(main_data, typeIndexPairs, indexAndType, ageDf, survivalAverages, avgAges, indexes, types, survivalRate, mostCommonEmbarkPont, possibleNames)

# we have to create train and test sets from the original train.csv (~30% test, ~70% train)
# the set is not in order so I will leave it as is
trainSet <- originalTrain[1:(nrow(originalTrain)*0.7), ]
testSet <- originalTrain[((nrow(originalTrain)*0.7)+1):(nrow(originalTrain)), ]

# the label is if the person survived or not
#trainLabels <- trainSet$Died
#testLabels <- testSet$Died

# deleting variables: Index, PassengerId, //Survived//, Name
# survived = 3
trainSet <- trainSet[, -c(1, 2, 5)]
testSet <- testSet[, -c(1, 2, 5)]

# MY ORIGINAL PLAN WAS TO USE A NEURAL NETWORK FOR THIS TASK, BUT DUE TO SOME ERRORS WITH PYTHON ENVIROMENTS
# I CREATED A FEW DAYS AGO, KERAS SHOWS ERRORS; I'M NOT SURE THAT HAVE TIME TO FIX IT BEFORE THE DEADLINE
# calling keras
# library(keras)
# 
# model = keras_model_sequential() %>% 
#   layer_dense(units=64, activation="relu", input_shape=7) %>% 
#   layer_dense(units=32, activation = "relu") %>% 
#   layer_dense(units=1, activation="sigmoid")
# 
# model %>% compile(
#   loss = "mse",
#   optimizer =  "adam", 
#   metrics = list("mean_absolute_error")
# )
# 
# model %>% summary()
# 
# model %>% fit(trainSet, trainLabels, epochs = 15)
# 
# scores = model %>% evaluate(testSet, testLabels)
# print(scores)

# SO FOR A QUICK SOLUTION I DECIDED TO USE A DECISION TREE WITH THE RPART LIBRARY
library(rpart)

# creating a decision tree, the variable we are predicting is death and the variables we use are the others in the df
fit <- rpart(Died ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Age, method = "anova", data = trainSet)

# quick summary
summary(fit)

# plot of the decision tree
#plot(fit, uniform = TRUE, 
#     main = "The Decision Tree:") 
#text(fit, use.n = TRUE, cex = .7)

# testing the tree with the test set
testSet$predictions <- predict(fit, testSet, method = "anova") 

# I decided to use simple MSE for evaluation, the model has a mean squared error of 0.125
MSE <- mean((testSet$Died - testSet$predictions)^2)

# I tested it with this package's MSE function, it returned the same value
#library(MLmetrics)
#MSE(testSet$predictions, testSet$Died)

# predicting chance of death in the official test csv
originalTest$ChanceOfDeath <- predict(fit, originalTest[, c(4,6,7,8,9,10,11)], method = "anova") 

# rebuilding into the form of the original csv
originalTestFull <- cbind(originalTest[, c(2,4,5,6,7,8,9,10,11,13)], test[, c(8,10)])

# reordering the variable columns
originalTestFull <- originalTestFull[, c(1,2,3,4,9,5,6,11,7,12,8,10)]

# the people with the highest chance of survival according to the model (or lowest chance of death)
peopleWhoProbablyLived <- data.frame(originalTestFull[which(originalTestFull$ChanceOfDeath == min(originalTestFull$ChanceOfDeath)), c(2,3,4,5)])

# creating the csv containing the Solution and filling it with the full dataframe
write.csv(originalTestFull,"Solution.csv", row.names = FALSE)
