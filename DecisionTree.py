import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import math

#Load dataset
housing_all = pd.read_csv('Housing.csv')

#Keep subset of features, and drop missing values
housing = housing_all[['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']]

# Encode binary and categorical features in the dataset
housing_data_encoded = housing.copy()
binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# Convert 'yes'/'no' to 1 and 0
for col in binary_columns:
    housing_data_encoded[col] = housing_data_encoded[col].apply(lambda x: 1 if x == 'yes' else 0)

# One-hot encode furnishingstatus
housing_data_encoded = pd.get_dummies(housing_data_encoded, columns=['furnishingstatus'], drop_first=True)

# Define features and target
X = housing_data_encoded.drop(columns=['price'])  # Feature matrix without target
y = housing_data_encoded['price']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a decision tree regressor model
tree = DecisionTreeRegressor(max_depth=10, min_samples_split=5, min_samples_leaf=5, random_state=42)
tree.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = tree.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
root_mse = math.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#R2
print("R-Squared: ", end="")
print("%.2f" % (100*r2))

#Mean Squared Error
print(f"Mean Squared Error: {mse:.2f}")

#Root Mean Squared Error
print(f"Root Mean Squared Error: ${root_mse:.2f}")

#Mean Absolute Error
print(f"Mean Absolute Error: ${mae:.2f}")
