import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


# Load data
housing = pd.read_csv('Housing.csv')

# Binary and categorical features in the dataset
housing_data_encoded = housing.copy()
binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# Convert 'yes'/'no' to 1 and 0 for features 
for col in binary_columns:
    housing_data_encoded[col] = housing_data_encoded[col].apply(lambda x: 1 if x == 'yes' else 0)

# # One-hot encode furnishingstatus
housing_data_encoded = pd.get_dummies(housing_data_encoded, columns=['furnishingstatus'], drop_first=True)

# Define input features and target
X = housing_data_encoded.drop(columns=['price'])  
y = housing_data_encoded['price'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Polynomial transformation x_train and X_test
poly_features = PolynomialFeatures(degree= 2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Scale X_train_poly and X_test_poly
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

ridge_model = Ridge(alpha=200.0)  # change for regularization strength
ridge_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = ridge_model.predict(X_test_scaled)


# Evaluate the model using Mean Squared Error and R-squared score
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
root_mse = math.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# Print
print("R-Squared: ", end="")
print("%.2f" % (100*r2))
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: ${root_mse:.2f}")
print(f"Mean Absolute Error: ${mae:.2f}")