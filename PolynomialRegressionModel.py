import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

housing_all = pd.read_csv('house-prices.csv')

housing = housing_all[['Home','Price','SqFt','Bedrooms','Bathrooms','Offers','Brick','Neighborhood']]


X = housing[['SqFt', 'Bedrooms']].values.reshape(-1, 2)
y = housing[['Price']].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 
poly_features = PolynomialFeatures(degree= 2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test_poly)

# Evaluate the model using Mean Squared Error and R-squared score
mse = mean_squared_error(y_test, y_pred)
r2_score = model.r2_score(X_test_poly, y_test)

print("Mean Squared Error:", mse)
print("R-squared:", r2_score * 100)

