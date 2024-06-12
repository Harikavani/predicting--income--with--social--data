# Install necessary packages if not already installed
packages <- c("tidyverse", "caret", "randomForest", "e1071", "ROCR")

install_if_missing <- function(p) {
  if (!require(p, character.only = TRUE)) {
    install.packages(p, dependencies = TRUE)
    library(p, character.only = TRUE)
  }
}

sapply(packages, install_if_missing)

# Load required libraries
library(tidyverse)
library(caret)
library(randomForest)
library(e1071)
library(ROCR)

# Download the dataset
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
download.file(url, destfile = "adult.data")

# Load the dataset
column_names <- c("age", "workclass", "fnlwgt", "education", "education_num", "marital_status", 
                  "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", 
                  "hours_per_week", "native_country", "income")

adult_data <- read.csv("adult.data", header = FALSE, col.names = column_names, na.strings = " ?")

# Explore the dataset
summary(adult_data)
str(adult_data)
head(adult_data)

# Handling missing values
adult_data <- adult_data %>% na.omit()

# Convert categorical variables to factors
adult_data <- adult_data %>% 
  mutate(across(where(is.character), as.factor))

# Splitting the data into training and testing sets
set.seed(123)
train_index <- createDataPartition(adult_data$income, p = 0.7, list = FALSE)
train_data <- adult_data[train_index, ]
test_data <- adult_data[-train_index, ]

# Train a Random Forest model
set.seed(123)
rf_model <- randomForest(income ~ ., data = train_data, importance = TRUE)

# Print the model
print(rf_model)

# Predictions on the test set
rf_predictions <- predict(rf_model, test_data)

# Confusion matrix
conf_matrix <- confusionMatrix(rf_predictions, test_data$income)
print(conf_matrix)

# ROC Curve
rf_prob <- predict(rf_model, test_data, type = "prob")
pred <- prediction(rf_prob[, 2], test_data$income)
perf <- performance(pred, "tpr", "fpr")
plot(perf, col = "blue", main = "ROC Curve")
abline(a = 0, b = 1, col = "red", lty = 2)

# AUC
auc <- performance(pred, "auc")
auc_value <- auc@y.values[[1]]
print(paste("AUC:", auc_value))

# Feature Importance Plot
varImpPlot(rf_model, main = "Feature Importance")

# Age vs Income
ggplot(adult_data, aes(x = age, fill = income)) + 
  geom_histogram(binwidth = 1, position = "dodge") + 
  labs(title = "Age vs Income", x = "Age", y = "Count")

# Hours per Week vs Income
ggplot(adult_data, aes(x = hours_per_week, fill = income)) + 
  geom_histogram(binwidth = 1, position = "dodge") + 
  labs(title = "Hours per Week vs Income", x = "Hours per Week", y = "Count")

# Workclass vs Income
ggplot(adult_data, aes(x = workclass, fill = income)) + 
  geom_bar(position = "dodge") + 
  labs(title = "Workclass vs Income", x = "Workclass", y = "Count")

# Education vs Income
ggplot(adult_data, aes(x = education, fill = income)) + 
  geom_bar(position = "dodge") + 
  labs(title = "Education vs Income", x = "Education", y = "Count") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Occupation vs Income
ggplot(adult_data, aes(x = occupation, fill = income)) + 
  geom_bar(position = "dodge") + 
  labs(title = "Occupation vs Income", x = "Occupation", y = "Count") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Sex vs Income
ggplot(adult_data, aes(x = sex, fill = income)) + 
  geom_bar(position = "dodge") + 
  labs(title = "Sex vs Income", x = "Sex", y = "Count")

# Marital Status vs Income
ggplot(adult_data, aes(x = marital_status, fill = income)) + 
  geom_bar(position = "dodge") + 
  labs(title = "Marital Status vs Income", x = "Marital Status", y = "Count") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

