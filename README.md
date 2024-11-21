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

```py

#Program to implement linear regression model for predicting car prices and test assumptions.
#Developed by: MOHAMMED SAAJID S
#RegisterNumber: 212223240093


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

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

# Standardize the features
SS = StandardScaler()

X = SS.fit_transform(X)
y = SS.fit_transform(y.values.reshape(-1, 1))



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
# 1. Linearity Assumption
plt.figure(figsize=(10, 6))
for i, col in enumerate(['horsepower', 'curbweight', 'enginesize', 'highwaympg']):
    plt.subplot(2, 2, i+1)
    plt.scatter(data[col], data['price'])
    plt.xlabel(col)
    plt.ylabel('Price')
    plt.title(f'Price vs {col}')
plt.tight_layout()
plt.show()

# 2. Homoscedasticity
plt.scatter(y_pred, y_test - y_pred)
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Prices')
plt.axhline(0, color='red', linestyle='--')
plt.show()

# 3. Normality
plt.figure(figsize=(10, 6))
plt.hist(y_test - y_pred, bins=30)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()

# 4. Multicollinearity
corr_matrix = data[['horsepower', 'curbweight', 'enginesize', 'highwaympg']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


```

## Output:

![alt text](image.png)

1. Linearity Assumption

![alt text](image-1.png)

2. Homoscedasticity

![alt text](image-2.png)

3. Normality

![alt text](image-3.png)

4. Multicollinearity

![alt text](image-4.png)

## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
