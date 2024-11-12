library(tidyverse)
library(caret)
library(ggplot2)
library(DataExplorer)
library(dplyr)
library(corrplot)
library(randomForest)
library(e1071)
library(GGally)

setwd("~/Documents/iris/")

# Read the dataset (using 'iris.data' as the data file)
data <- read.csv("iris.data", header = FALSE)

# Assign column names as per the iris dataset features
colnames(data) <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species")

# Step 1: Initial Data Overview
print("Initial Dataset Overview")
print(str(data))
print(summary(data))
print(dim(data))

# Step 2: Checking for Missing Values
missing_counts <- colSums(is.na(data))
print("Missing Value Counts:")
print(missing_counts)

# Visualize Missing Values
plot_missing(data)

# Step 3: Data Cleaning
# Impute Missing Values (if any exist)
# Use median for numerical columns, mode for categorical columns
data <- data %>%
  mutate_if(is.numeric, ~ifelse(is.na(.), median(., na.rm = TRUE), .)) %>%
  mutate_if(is.character, ~ifelse(is.na(.), Mode(.), .))

# Convert categorical variables to factors
data <- data %>%
  mutate_if(is.character, as.factor)

# Step 4: Outlier Detection and Removal
# Using the Interquartile Range (IQR) method
detect_outliers <- function(x) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  x > (Q3 + 1.5 * IQR) | x < (Q1 - 1.5 * IQR)
}

outliers <- data %>%
  select_if(is.numeric) %>%
  map(detect_outliers)

print("Outliers Detected in Numerical Columns:")
print(sapply(outliers, sum))

# Removing rows with extreme outliers
data <- data[!rowSums(as.data.frame(outliers)), ]

# Step 5: Feature Scaling (Standardization)
# Scale numerical columns to have mean = 0 and sd = 1
data <- data %>%
  mutate_if(is.numeric, scale)

# Step 6: Exploratory Data Analysis (EDA)
# Distribution of each variable
print("Data Distributions:")
plot_histogram(data)
plot_density(data)

# Pairwise Relationships (Visualizing correlations between numerical features)
print("Pairwise Relationships:")
ggpairs(data, aes(color = Species))

# Correlation Analysis for Numerical Columns
if (ncol(data %>% select_if(is.numeric)) > 1) {
  corr_matrix <- cor(data %>% select_if(is.numeric))
  print("Correlation Matrix:")
  print(corr_matrix)
  corrplot(corr_matrix, method = "ellipse", type = "upper", order = "hclust")
}

# Checking Class Distribution (for classification problems)
if (is.factor(data[[ncol(data)]])) {
  print("Class Distribution:")
  print(table(data[[ncol(data)]]))
}

# Step 7: Splitting the Dataset into Training and Testing Sets
set.seed(123)
trainIndex <- createDataPartition(data[[ncol(data)]], p = 0.7, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]

# Step 8: Model Selection Based on Problem Type
if (is.factor(data[[ncol(data)]])) {
  print("Problem Type: Classification")
  models <- c("rpart", "rf", "gbm", "knn", "naive_bayes")
} else {
  print("Problem Type: Regression")
  models <- c("lm", "rf", "gbm", "svmRadial", "ridge")
}

# Step 9: Model Training and Evaluation
set.seed(123)
train_control <- trainControl(method = "cv", number = 5, savePredictions = TRUE)

# Model Evaluation Results Storage
model_results <- list()

# Train and Evaluate Models
for (model in models) {
  print(paste("Training model:", model))
  
  # Train model using caret
  fit <- train(
    as.formula(paste(names(data)[ncol(data)], "~ .")),
    data = train,
    method = model,
    trControl = train_control
  )
  
  # Model Predictions
  predictions <- predict(fit, newdata = test)
  
  # Evaluate the Model
  if (is.factor(data[[ncol(data)]])) {
    # For Classification - Confusion Matrix, Accuracy, F1 Score
    confusion <- confusionMatrix(predictions, test[[ncol(data)]])
    accuracy <- confusion$overall["Accuracy"]
    f1_score <- confusion$byClass["F1"]
    model_results[[model]] <- list(Model = model, Accuracy = accuracy, F1 = f1_score)
  } else {
    # For Regression - RMSE, R-squared
    rmse <- RMSE(predictions, test[[ncol(data)]])
    rsq <- R2(predictions, test[[ncol(data)]])
    model_results[[model]] <- list(Model = model, RMSE = rmse, R2 = rsq)
  }
}

# Display Model Results
print("Model Evaluation Results:")
print(model_results)

# Step 10: Recommending the Best Model
if (is.factor(data[[ncol(data)]])) {
  best_model <- model_results[which.max(sapply(model_results, function(x) x$Accuracy))]
  print("Best Classification Model Based on Accuracy:")
} else {
  best_model <- model_results[which.min(sapply(model_results, function(x) x$RMSE))]
  print("Best Regression Model Based on RMSE:")
}
print(best_model)
