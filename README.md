# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Libraries**: Bring in essential libraries such as pandas, numpy, matplotlib, and sklearn.
2. **Load Dataset**: Import the dataset containing car prices along with relevant features.
3. **Data Preprocessing**: Manage missing data and select key features for the model, if required.
4. **Split Data**: Divide the dataset into training and testing subsets.
5. **Train Model**: Build a linear regression model and train it using the training data.
6. **Make Predictions**: Apply the model to predict outcomes for the test set.
7. **Evaluate Model**: Measure the model's performance using metrics like R² score, Mean Absolute Error (MAE), etc.
8. **Check Assumptions**: Plot residuals to verify assumptions like homoscedasticity, normality, and linearity.
9. **Output Results**: Present the predictions and evaluation metrics.

## Program:
```
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: Aaron H 
RegisterNumber: 212223040001
*/
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset from the URL
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv")

# Display the first few rows of the dataset
print(data.head())

# Data Preprocessing
# Handle missing values (if any)
data = data.dropna()  # Drop rows with missing values

# Select features and target variable
# Assume 'price' is the target variable and 'horsepower', 'curbweight', 'enginesize', and 'highwaympg' are features
X = data[['horsepower', 'curbweight', 'enginesize', 'highwaympg']]
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Check model assumptions
plt.scatter(y_pred, y_test - y_pred)
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Prices')
plt.axhline(0, color='red', linestyle='--')
plt.show()
```

## Output:
<img width="803" alt="Screenshot 2024-10-06 at 8 41 08 PM" src="https://github.com/user-attachments/assets/7c99f174-3c6a-4c28-929e-e9e5f2ea84cb">
<img width="803" alt="Screenshot 2024-10-06 at 8 41 13 PM" src="https://github.com/user-attachments/assets/bdfe064d-b1ae-43dd-a96d-553c507605ee">



## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
