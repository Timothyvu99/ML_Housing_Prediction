import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# Load the dataset
housing_data = pd.read_csv('Housing.csv')

# Keep relevant features and manually encode binary/categorical variables
housing = housing_data[['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']]

# Encode binary categorical features ('yes'/'no' to 1/0)
binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                  'airconditioning', 'prefarea']
for col in binary_columns:
    housing[col] = housing[col].apply(lambda x: 1 if x == 'yes' else 0)

# One-hot encode furnishingstatus
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
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr.fit(X_train_scaled, y_train)

# Make prediction and evaluate the model
y_pred = svr.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
root_mse = math.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print
print("R2: ", end="")
print("%.2f" % (100*r2))
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: ${root_mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
