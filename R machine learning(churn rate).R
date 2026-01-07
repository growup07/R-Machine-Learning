# Yeong Siang Chiam, Foram Ram Nagare, Maddie Nowaczyk, Manikanta Reddy Pingali,
# Kate Reisslein
# MIS 545
# GroupProjectGroup1.R
# The purpose ofthis code is to look at customer churn data, perform data 
# preprocessing and 4 different modeling approaches (logistic regression,
# k-nearest neighbors, naive bayes, and decision tree)

# Install the needed packages
# install.packages("tidyverse")
# install.packages("corrplot")
# install.packages("olsrr")
# install.packages("smotefamily")
# install.packages("class")
# install.packages("rpart.plot")
# install.packages("e1071")

# loading libraries
library(tidyverse)
library(corrplot)
library(dplyr)
library(olsrr)
library(smotefamily)
library(class)
library(rpart)
library(rpart.plot)
library(e1071)


# Data Preprocessing ------------------------------------------------------

# Setting working Directory
setwd("C:/Users/maddi/OneDrive/Desktop/MIS 545/Group Project")

# Reading data into a churn tibble
churnBinned <- read_csv(file="churnBinned.csv",
                  col_types = "lillnininnnnnnnnnnn",
                  col_names=TRUE)

# Seperate binned data from normal data
churn <- churnBinned %>%
  select(Churn, AccountWeeks, ContractRenewal, DataPlan, DataUsage,
         CustServCalls, DayMins, DayCalls, MonthlyCharge, OverageFee, RoamMins)
churnBinnedData <- churnBinned %>%
  select(Churn, ContractRenewal, DataPlan, AccountWeeksBinned, DataUsageBinned,
         CustServCallsBinned, DayMinsBinned, DayCallsBinned, MonthlyChargeBinned,
         OverageFeeBinned, RoamMinsBinned)

# Displaying structure of churn
str(churn)

# Displaying the summary of churn
summary(churn)

# Displaying the correlation 
cor(churn %>% keep(is.numeric))

# Rounding the correlation to 2 decimals
round(cor(churn),2)

# Displaying Correalation matrix
corrplot(cor(churn),method ="number", type ="lower")

# Removing Monthlycharges and DataUsage from data as highly corelated
churn <- churn %>% select(-MonthlyCharge, -DataUsage)
churnBinnedData <- churnBinnedData %>% select(-MonthlyChargeBinned,
                                              -DataUsageBinned)

# Displaying Correalation matrix
corrplot(cor(churn),method ="number", type ="lower")

# Displaying Histogram using displayAllHistogram() function
displayAllHistograms <- function(tibbleDataset) {
  tibbleDataset %>%
    keep(is.numeric) %>%
    gather() %>%
    ggplot() + geom_histogram(mapping = aes(x=value,fill=key),
                              color = "black") +
    facet_wrap(~ key, scales = "free") + theme_minimal() }

# Calling the displayAllHistogram() function
displayAllHistograms(churn)

# Determining outliers in the CustServCalls feature
Q1 <- quantile(churn$CustServCalls, 0.25)
Q3 <- quantile(churn$CustServCalls, 0.75)
IQR <- Q3 - Q1
outlierMin <- Q1 - 1.5 * IQR
print(outlierMin)
outlierMax <- Q3 + 1.5 * IQR
print(outlierMax)

# Keep the outliers in the dataset, but add the outliers to their own tibble
# called churnOutliers
churnOutliers <- churn %>%
  filter(churn$CustServCalls < outlierMin | churn$CustServCalls > outlierMax)

# Normalize the CustServCalls feature by taking the log of CustServCalls into a 
# new feature called logCustServCalls
# churn <- churn %>%
  # mutate(logCustServCalls = log(CustServCalls))

# Calling the displayAllHistogram() function after transforming 
# CustServCalls using log
displayAllHistograms(churn)

# Displaying boxplot compares the number of custServCalls and churn
ggplot(churn, aes(x = factor(Churn), y = CustServCalls)) + geom_boxplot()

# Displaying boxplot compares the RoamMins and churn
ggplot(churn, aes(x = factor(Churn), y = RoamMins)) + geom_boxplot()

# Count churn rate for customers with >4 customer service calls
churn %>%
  filter(CustServCalls > 4) %>%
  summarize(churn_rate = mean(Churn)) 

# Compare churn rate for customers who renewed vs did not renew contract
churn %>%
  group_by(ContractRenewal) %>%
  summarize(churn_rate = mean(Churn))

# Define high roaming as above median and calculate churn rate
median_roam <- median(churn$RoamMins)
churn %>%
  mutate(high_roaming = RoamMins > median_roam) %>%
  group_by(high_roaming) %>%
  summarize(churn_rate = mean(Churn))

# Logistic Regression -----------------------------------------------------

# Split data into training and testing sets.
# set.seed() ensures reproducible results for random sampling.
set.seed(545)

# Randomly select 75% of row indices from the dataset.
sampleSet <- sample(nrow(churn),
                    round(nrow(churn) * 0.75), 
                    replace = FALSE)

# Assign the 75% sampled rows to churnTraining.
# Training set has 2,500 observations.
churnTraining <- churn[sampleSet, ]

# Assign the remaining 25% of rows to churnTesting.
# Testing set contains 833 observations.
churnTesting <- churn[-sampleSet, ]

# Do we have class imbalance in data? Yes we do.
summary(churnTraining$Churn)

# Store the magnitude of class imbalance into a variable. 
classImbalanceMagnite <- 2145 / 355
classImbalanceMagnite

# Deal with class imbalance in the training dataset using SMOTE Technique,
# to generate synthetic records, closer to 50:50 splits btw 1 & 0 .
churnTrainingSmoted <- tibble(SMOTE(X = data.frame(
  churnTraining), target = churnTraining$Churn,
  dup_size = 5)$data)

# The Dup_size parameter only accepts real numbers. 
# We can adjust it iteratively to reach a mean churn value close to 0.50.
# Our current mean churn value is 0.4981, 
# which is very close to the desired 0.50 balance.
summary(churnTrainingSmoted)

# Convert columns back to logical after SMOTE changed them to numeric.
churnTrainingSmoted <- churnTrainingSmoted %>% 
  mutate(Churn = as.logical(Churn),
         ContractRenewal = as.logical(ContractRenewal), 
         DataPlan = as.logical(DataPlan))

# Columns returned to logical. 
summary(churnTrainingSmoted)

# Drop the "class" column added by SMOTE() from the tibble.
churnTrainingSmoted <- churnTrainingSmoted %>% 
  select(-class)

# Class column drop. 
summary(churnTrainingSmoted)

# Fit a logistic regression model using glm() with a binomial outcome.
# Predicting Withdraw using all available predictors.
churnModel <- glm(data = churnTrainingSmoted, 
                  family = binomial, formula = Churn ~ .)

# Significant parameter as below: 
# ContractRenewalTRUE, CustServCalls, DayMins, RoamMins.
summary(churnModel)

# Coefficients (Estimates) are in log-odds, which are hard to interpret.
# Convert them to odds ratios for easier interpretation:
# - Odds ratio > 1: increase in predictor increases the odds of churn
# - Odds ratio < 1: increase in predictor decreases the odds of churn
# - Odds ratio = 1: predictor has no effect on the odds of churn
# Always check the p-values to assess significance, 
# even after examining coefficients
exp(coef(churnModel)["AccountWeeks"])
exp(coef(churnModel)["ContractRenewalTRUE"])
exp(coef(churnModel)["DataPlanTRUE"])
exp(coef(churnModel)["CustServCalls"])
exp(coef(churnModel)["DayMins"])
exp(coef(churnModel)["DayCalls"])
exp(coef(churnModel)["OverageFee"])
exp(coef(churnModel)["RoamMins"])

# Use the model to predict outcomes on the test dataset.
# Output: 833 values between 0 and 1, representing predicted.
# probability of withdrawal
# - < 0.5 → predicted not to churn
# - ≥ 0.5 → predicted to churn
churnPrediction <- predict(churnModel, 
                           churnTesting, type = "response")

# Display churnPrediction on the console.
print(churnPrediction)

# Convert probabilities to binary outcomes: ≤0.5 → 0, >0.5 → 1 using ifelse.
churnPrediction <- ifelse(churnPrediction >= 0.5, 1, 0)

# Display churnPrediction on the console.
print(churnPrediction)

# Create confusion matrix.
churnConfusionMatrix <- table(churnTesting$Churn,
                              churnPrediction)

# Display churnConfusionMatrix on the console.
print(churnConfusionMatrix)

# Calculate the false negative rate:
churnConfusionMatrix[2,1] / 
  (churnConfusionMatrix[2,1] + 
     churnConfusionMatrix[2,2])

# Calculate the false positive rate:
churnConfusionMatrix[1,2] / 
  (churnConfusionMatrix[1,2] + 
     churnConfusionMatrix[1,1])

# Finally, calculate prediction accuracy:
# (Number of true positives + true negatives) / total predictions
# Result: 70.0% accuracy on the test dataset.
sum(diag(churnConfusionMatrix)) / nrow(churnTesting)

# k-nearest neighbors -----------------------------------------------------

# Separate the tibble into two. One with just the label and one with the other 
# variables. 
churnLabels <- churn %>%
  select(Churn)
churn2 <- churn %>%
  select(-Churn)

# Randomly split the dataset into churnTraining (75% of records) and 
# churnTesting (25% of records) using 545 as the random seed 
set.seed(545)
sampleSet <- sample(nrow(churn2),
                    round(nrow(churn2) * 0.75),
                    replace = FALSE)

# Put the records from the 75% sample into churnTraining (3 variables) and 
# churnTrainingLabels (1 variable). Hint: Watch the video for clarification 
# on how this is accomplished
churnTraining2 <- churn2[sampleSet, ]
churnTrainingLabels <- churnLabels[sampleSet, ]

# Put the remaining 25% of records into churnTesting (3 variables) and 
# churnTestingLabels (1 variable). Hint: Watch the video for clarification 
# on how this is accomplished.
churnTesting2 <- churn2[-sampleSet, ]
churnTestingLabels <- churnLabels[-sampleSet, ]

# Generate the k-nearest neighbors model using churnTraining as the train 
# argument, churnTesting as the test argument, 
# churnTrainingLabels$Churn as the cl argument, and 49 as the value for 
# the k argument. Note: We are using 49 as it is the nearest odd number to the 
# square root of records in the training dataset Now using 51 because it is the
# best predictive accuaracy value. 
churnPrediction <- knn(train = churnTraining2,
                       test = churnTesting2,
                       cl = churnTrainingLabels$Churn,
                       k = 51)

# Display the predictions from the testing dataset on the console
print(churnPrediction)

# Display summary of the predictions from the testing dataset
print(summary(churnPrediction))

# Evaluate the model by forming a confusion matrix
churnConfusionMatrix <- table(churnTestingLabels$Churn,
                              churnPrediction)

# Display the confusion matrix on the console
print(churnConfusionMatrix)

# Calculate false positive rate
churnConfusionMatrix[1, 2] / 
  (churnConfusionMatrix[1, 2] +
     churnConfusionMatrix[1, 1])

# Calculate false negative rate
churnConfusionMatrix[2, 1] / 
  (churnConfusionMatrix[2, 1] +
     churnConfusionMatrix[2, 2])

# Calculate the model predictive accuracy and store it into a variable called 
# predictiveAccuracy
predictiveAccuracy <- sum(diag(churnConfusionMatrix)) /
  nrow(churnTesting)

# Display the predictive accuracy on the console
print(predictiveAccuracy)

# Create a matrix of k-values with their predictive accuracy (the matrix will be
# empty and have 2 columns and 0 rows). Store the matrix into an object called 
# kValueMatrix. 
kValueMatrix <- matrix(data = NA,
                       nrow = 0,
                       ncol = 2)

# Assign column names of "k value" and "Predictive accuracy" to the kValueMatrix.
colnames(kValueMatrix) <- c("k value", "Predictive accuracy")

# Loop through odd values of k from 1 up to the number of records in the training
# dataset. With each pass through the loop, store the k-value along with its 
# predictive accuracy. Hint: Watch the video for clarification on how this is 
# accomplished. 
for (kValue in 1:nrow(churnTraining2)) {
  if(kValue %% 2 !=0) {
    churnPrediction <- knn(train = churnTraining2,
                           test = churnTesting2,
                           cl = churnTrainingLabels$Churn,
                           k = kValue)
    churnConfusionMatrix <- table(churnTestingLabels$Churn,
                                  churnPrediction)
    predictiveAccuracy <- sum(diag(churnConfusionMatrix)) /
      nrow(churnTesting2)
    kValueMatrix <- rbind(kValueMatrix, c(kValue, predictiveAccuracy))
  }
}

# Display the kValueMatrix on the console to determine the best k-value
print(kValueMatrix)

# Naive Bayes -------------------------------------------------------------
# reading logical columns as factors for Naive Bayes Model
churnBinnedData$ContractRenewal <- as.factor(churnBinned$ContractRenewal)
churnBinnedData$DataPlan        <- as.factor(churnBinned$DataPlan)
churnBinnedData$Churn           <- as.factor(churnBinned$Churn)

# Display the churnBinned dataset on the console
print(churnBinnedData)

# Displaying structure of churnBinned dataset
str(churnBinnedData)

# Displaying the summary of churnBinned dataset
summary(churnBinnedData)

# Randomly split the dataset into churnBinnedTraining (75% of record
# churnBinnedTesting (25% of records) using 545 as the random seed 
set.seed(545)
sampleSet <- sample(nrow(churnBinnedData),
                    round(nrow(churn) * 0.75),
                    replace = FALSE)
churnBinnedTraining <- churnBinnedData[sampleSet, ]
churnBinnedTesting <- churnBinnedData[-sampleSet, ]

# Build the Naïve Bayes Model using Training Data
churnBinnedModel <- naiveBayes(formula = Churn ~.,
                               data = churnBinnedTraining,
                               laplace = 1)

# Predict Probabilities for the Testing Data
churnBinnedProbability <- predict(churnBinnedModel,churnBinnedTesting, 
                                  type = "raw")

# Display churnBinnedProbability on the console
print(churnBinnedProbability)

# Predict classes for each record in the testing dataset
churnBinnedPrediction <- predict(churnBinnedModel,churnBinnedTesting, 
                                 type = "class")

# Display churnBinnedPrediction on the console
print(churnBinnedPrediction)

# Evaluate the model by forming a confusion matrix
churnBinnedConfusionMatrix <- table(churnBinnedTesting$Churn, 
                                    churnBinnedPrediction)

# Display the confusion matrix on the console
print(churnBinnedConfusionMatrix)

# false positive rate
churnBinnedConfusionMatrix[1, 2] / 
  (churnBinnedConfusionMatrix[1, 2] + 
     churnBinnedConfusionMatrix[1, 1])

# false negative rate
churnBinnedConfusionMatrix[2, 1] / 
  (churnBinnedConfusionMatrix[2, 1] + 
     churnBinnedConfusionMatrix[2, 2])

# Calculate the model predictive accuracy 
predictiveAccuracy <- sum(diag(churnBinnedConfusionMatrix)) / 
  nrow(churnBinnedTesting)

# Display the predictive accuracy on the console
print(predictiveAccuracy)

# Decision Tree -----------------------------------------------------------

# split data into train and test 
set.seed(545)
sampleSet <- sample(nrow(churn), round(nrow(churn)) * 0.75, 
                    replace = FALSE)
churnTraining <- churn[sampleSet, ]
churnTesting <- churn[-sampleSet, ]


# cp = 0.01 ---------------------------------------------------------------

# Generate the decision tree model to predict customer churn based on the other 
# variables in the dataset. Use 0.01 as the complexity parameter.
churnDecisionTreeModel <- rpart(formula = Churn ~ .,
                                method = "class",
                                cp = 0.01,
                                data = churnTraining)

# Display the decision tree visualization in R
rpart.plot(churnDecisionTreeModel)

# Predict classes for each record in the testing dataset
churnPrediction <- predict(churnDecisionTreeModel,
                           churnTesting,
                           type = "class")

# Display churnPrediction on the console
print(churnPrediction)

# Evaluate the model by forming a confusion matrix
churnConfusionMatrix <- table(churnTesting$Churn,
                              churnPrediction)

# Display the confusion matrix on the console
print(churnConfusionMatrix)

# false positive rate
churnConfusionMatrix[1, 2] / 
  (churnConfusionMatrix[1, 2] + 
     churnConfusionMatrix[1, 1])

# false negative rate
churnConfusionMatrix[2, 1] / 
  (churnConfusionMatrix[2, 1] + 
     churnConfusionMatrix[2, 2])

# Calculate the model predictive accuracy 
predictiveAccuracy <- sum(diag(churnConfusionMatrix)) / 
  nrow(churnTesting)

# Display the predictive accuracy on the console
print(predictiveAccuracy)


# cp = 0.001 --------------------------------------------------------------

# Generate the decision tree model to predict customer churn based on the other 
# variables in the dataset. Use 0.001 as the complexity parameter.
churnDecisionTreeModel2 <- rpart(formula = Churn ~ .,
                                 method = "class",
                                 cp = 0.001,
                                 data = churnTraining)

# Display the decision tree visualization in R
rpart.plot(churnDecisionTreeModel2)

# Predict classes for each record in the testing dataset
churnPrediction2 <- predict(churnDecisionTreeModel2,
                            churnTesting,
                            type = "class")

# Display churnPrediction on the console
print(churnPrediction2)

# Evaluate the model by forming a confusion matrix
churnConfusionMatrix2 <- table(churnTesting$Churn,
                               churnPrediction2)

# Display the confusion matrix on the console
print(churnConfusionMatrix2)

# false positive rate
churnConfusionMatrix2[1, 2] / 
  (churnConfusionMatrix2[1, 2] + 
     churnConfusionMatrix2[1, 1])

# false negative rate
churnConfusionMatrix2[2, 1] / 
  (churnConfusionMatrix2[2, 1] + 
     churnConfusionMatrix2[2, 2])

# Calculate the model predictive accuracy 
predictiveAccuracy2 <- sum(diag(churnConfusionMatrix2)) / 
  nrow(churnTesting)

# Display the predictive accuracy on the console
print(predictiveAccuracy2)


