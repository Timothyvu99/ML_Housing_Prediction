# Import necessary libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np

# Load the dataset
housing = pd.read_csv('Housing.csv')

# Encode binary and categorical features in the dataset
housing_data_encoded = housing.copy()
binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# Convert 'yes'/'no' to 1 and 0
for col in binary_columns:
    housing_data_encoded[col] = housing_data_encoded[col].apply(lambda x: 1 if x == 'yes' else 0)

# One-hot encode furnishingstatus
housing_data_encoded = pd.get_dummies(housing_data_encoded, columns=['furnishingstatus'], drop_first=True)

# Define price categories (low, medium, high) based on percentiles
housing_data_encoded['price_category'] = pd.qcut(housing['price'], q=3, labels=['low', 'medium', 'high'])

# Define input features and target variable
X = housing_data_encoded.drop(columns=['price_category']) # Example feature set
y = housing_data_encoded['price_category']

# Label encode the target variable (for numeric compatibility)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Scale input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the Naive Bayes model
naive_bayes_model = GaussianNB()

# Fit the model to the entire dataset
naive_bayes_model.fit(X_scaled, y)

# Predict on the entire dataset
y_pred = naive_bayes_model.predict(X_scaled)

# Calculate performance metrics
accuracy = accuracy_score(y, y_pred)
conf_matrix = confusion_matrix(y, y_pred)
classification_rep = classification_report(y, y_pred)

# Regression metrics
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Print results
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)

# Print
print(f"R2: {r2*100:.2f}")
print(f"\nMean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: ${rmse:.2f}")
print(f"Mean Absolute Error: ${mae:.2f}")
