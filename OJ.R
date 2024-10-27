# Load necessary libraries
library(ISLR2)
library(caret)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(dplyr)

# Load the OJ dataset
data(OJ)

# Check for missing values (none in this dataset)
sum(is.na(OJ))

# Convert Purchase to a factor (if not already)
OJ$Purchase <- as.factor(OJ$Purchase)

# Split the data into training (70%) and testing (30%) sets
set.seed(123)
train_index <- createDataPartition(OJ$Purchase, p = 0.7, list = FALSE)
train_data <- OJ[train_index, ]
test_data <- OJ[-train_index, ]

# Exploratory Data Analysis (EDA) with additional plots
## 1. Distribution of Purchase
ggplot(OJ, aes(x = Purchase)) +
  geom_bar(fill = "lightblue") +
  ggtitle("Distribution of Purchase (Citrus Hill vs. Minute Maid)") +
  xlab("Purchase") + ylab("Count")

## 2. Scatter plot for PriceCH vs PriceMM (key variables)
ggplot(OJ, aes(x = PriceCH, y = PriceMM, color = Purchase)) +
  geom_point(alpha = 0.6) +
  ggtitle("Price of Citrus Hill vs. Price of Minute Maid by Purchase") +
  xlab("Price of Citrus Hill") + ylab("Price of Minute Maid")

## 3. Distribution plot for PriceCH and PriceMM
ggplot(OJ, aes(x = PriceCH, fill = Purchase)) +
  geom_density(alpha = 0.4) +
  ggtitle("Distribution of Price for Citrus Hill by Purchase") +
  xlab("Price of Citrus Hill")

ggplot(OJ, aes(x = PriceMM, fill = Purchase)) +
  geom_density(alpha = 0.4) +
  ggtitle("Distribution of Price for Minute Maid by Purchase") +
  xlab("Price of Minute Maid")

# Build the initial Decision Tree model
tree_model <- rpart(Purchase ~ ., data = train_data, method = "class")

# Plot the Decision Tree
rpart.plot(tree_model, type = 3, extra = 102, under = TRUE, fallen.leaves = TRUE)

# Evaluate the initial model on the test data
test_pred <- predict(tree_model, newdata = test_data, type = "class")
confusionMatrix(test_pred, test_data$Purchase)

# Calculate initial accuracy
initial_accuracy <- sum(test_pred == test_data$Purchase) / nrow(test_data)
print(paste("Initial Accuracy:", round(initial_accuracy * 100, 2), "%"))

# Feature Importance Plot
var_imp <- as.data.frame(varImp(tree_model, scale = FALSE))
var_imp$Variable <- rownames(var_imp)
ggplot(var_imp, aes(x = reorder(Variable, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  xlab("Features") + ylab("Importance") +
  ggtitle("Feature Importance in Initial Decision Tree")

# Tune the Decision Tree (Cross-validation for optimal cp)
set.seed(123)
tree_tuned <- train(Purchase ~ ., data = train_data, method = "rpart",
                    trControl = trainControl(method = "cv", number = 10),
                    tuneLength = 10)

# Plot the tuned model's performance across cp values
plot(tree_tuned)
print(tree_tuned$bestTune)

# Final Decision Tree after tuning
final_tree_model <- rpart(Purchase ~ ., data = train_data, method = "class", 
                          control = rpart.control(cp = tree_tuned$bestTune$cp))

# Plot the improved Decision Tree
rpart.plot(final_tree_model, type = 3, extra = 102, under = TRUE, fallen.leaves = TRUE, 
           main = "Improved Decision Tree")

# Evaluate the improved model on the test data
final_test_pred <- predict(final_tree_model, newdata = test_data, type = "class")
confusionMatrix(final_test_pred, test_data$Purchase)

# Calculate final accuracy
final_accuracy <- sum(final_test_pred == test_data$Purchase) / nrow(test_data)
print(paste("Final Accuracy:", round(final_accuracy * 100, 2), "%"))
