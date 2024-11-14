from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import pandas as pd
import numpy as np
import math

# Load data
housing = pd.read_csv('Housing.csv')

# Encode binary and categorical features in the dataset
housing_data_encoded = housing.copy()
binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# Convert 'yes'/'no' to 1 and 0
for col in binary_columns:
    housing_data_encoded[col] = housing_data_encoded[col].apply(lambda x: 1 if x == 'yes' else 0)

# # One-hot encode furnishingstatus
housing_data_encoded = pd.get_dummies(housing_data_encoded, columns=['furnishingstatus'], drop_first=True)

# Define input features and target
X = housing_data_encoded.drop(columns=['price'])  
# X = housing_data_encoded[['area', 'bedrooms', 'bathrooms']]
y = housing_data_encoded['price']  

# Polynomial Features
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.4, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100],
    'max_depth': [20],
    'min_samples_split': [5],
    'min_samples_leaf': [2]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Best model and predictions
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
root_mse = math.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#Print best hyperparameter
print(f"Best hyperparameter: {grid_search.best_params_}")

#R2
print("Accuracy (r2): ", end="")
print("%.2f" % (100*r2))

#Mean Squared Error
print(f"Mean Squared Error: {mse}")

#Root Mean Squared Error
print(f"Root Mean Squared Error: ${root_mse:.2f}")

#Mean Absolute Error
print(f"Mean Absolute Error: ${mae:.2f}")