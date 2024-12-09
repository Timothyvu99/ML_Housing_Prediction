# Import packages and functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import math


# Load the dataset
housing_all = pd.read_csv('Housing.csv')

# Keep subset of features, and drop missing values
housing = housing_all[['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']]

# Encode binary and categorical features in the dataset
housing_data_encoded = housing.copy()
binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# Convert 'yes'/'no' to 1 and 0
for col in binary_columns:
    housing_data_encoded[col] = housing_data_encoded[col].apply(lambda x: 1 if x == 'yes' else 0)

# # One-hot encode furnishingstatus
housing_data_encoded = pd.get_dummies(housing_data_encoded, columns=['furnishingstatus'], drop_first=True)

# Define features and target
X = housing_data_encoded.drop(columns=['price']) 
y = housing_data_encoded['price']  

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a multiple linear regression model
multRegMod = LinearRegression()
multRegMod.fit(X_train_scaled, y_train)

# Make predictions and evaluate the model
y_pred = multRegMod.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
root_mse = math.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# R2
print("R-Squared: ", end="")
print("%.2f" % (100*r2))

# Mean Squared Error
print(f"Mean Squared Error: {mse:.2f}")

# Root Mean Squared Error
print(f"Root Mean Squared Error: ${root_mse:.2f}")

# Mean Absolute Error
print(f"Mean Absolute Error: ${mae:.2f}")
