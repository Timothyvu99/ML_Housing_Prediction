import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
housing_data = pd.read_csv('Housing.csv')

# Keep relevant features and manually encode binary/categorical variables
housing = housing_data[['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']]

# Encode binary categorical features ('yes'/'no' to 1/0)
binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                  'airconditioning', 'prefarea']
for col in binary_columns:
    housing[col] = housing[col].apply(lambda x: 1 if x == 'yes' else 0)

# One-hot encode 'furnishingstatus' without using a pipeline
housing = pd.get_dummies(housing, columns=['furnishingstatus'], drop_first=True)

# Define features and target
X = housing.drop(columns=['price']) 
y = housing['price'] 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Support Vector Regressor
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)

# Make predictions and evaluate the model
y_pred = svr_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output results
print("Support Vector Regression Results:")
print("R^2 Score:", f"{r2:.2f}")
print("Mean Squared Error:", f"{mse:.2f}")
print("Mean Absolute Error:", f"{mae:.2f}")
