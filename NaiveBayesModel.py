# Import necessary packages and functions
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
housing_all = pd.read_csv('Housing.csv')

# Keep subset of features and drop missing values
housing = housing_all[['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']].dropna()

# Define price categories (low, medium, high) based on percentiles
housing['price_category'] = pd.qcut(housing['price'], q=3, labels=['low', 'medium', 'high'])

# Define input features (area and bedrooms) and target variable (price category)
x = housing[['area', 'bedrooms']].values
y = housing['price_category'].values

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

# Initialize the Gaussian Naive Bayes model
naive_bayes_model = GaussianNB()

# Fit the model to the training data
naive_bayes_model.fit(x_train, y_train)

# Predict on the test data
y_pred = naive_bayes_model.predict(x_test)

# Calculate accuracy and display performance metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Display classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Example prediction
example_data = np.array([[2500, 3]])  # Example: Area = 2500, Bedrooms = 3
predicted_category = naive_bayes_model.predict(example_data)
print(f"Predicted Price Category for example data {example_data}: {predicted_category[0]}")
