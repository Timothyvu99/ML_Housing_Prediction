# Import packages and functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm

#Load the dataset
housing_all = pd.read_csv('Housing.csv')

#Keep subset of features, and drop missing values
housing = housing_all[['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']]

#Define input and output features
x = housing[['area', 'bedrooms']].values.reshape(-1, 2)
y = housing[['price']].values.reshape(-1, 1)

#Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, random_state = 42)

#initialize multiple linear regression model
multRegMod = LinearRegression()

#Fit multiple linear regression model
multRegMod.fit(x_train, y_train)

#Print intercept and weight
print(f"Multiple Regression Model Intercept: {multRegMod.intercept_}")
print(f"Multiple Regression Model Weight: {multRegMod.coef_}")

#Predicted and Residual
yPredicted = multRegMod.predict(x_train)
residual = y_train - yPredicted

print(f"Predicted Price: {yPredicted[:6]}")
print(f"Residual: {residual[:6]}")

#Accuracy
accuracy = multRegMod.score(x, y)

print("Accuracy: ", end="")
print("%.2f" % (100*accuracy))

#Adjusted R-squared
n = x_train.shape[0]
p = x_train.shape[1]
adjusted_accuracy = 1 - (1-accuracy) * (n - 1) / (n - p - 1)
print("Adjusted Accuracy: ", end="")
print("%.2f" % (100*adjusted_accuracy))

#Mean Squared Error
mse = mean_squared_error(y_train, yPredicted)
print(f"Mean Squared Error: {mse}")

#Mean Absolute Error
mae = mean_absolute_error(y_train, yPredicted)
print(f"Mean Absolute Error: {mae}")

x_train_const = sm.add_constant(x_train)
sm_model = sm.OLS(y_train, x_train_const).fit()
print(f"P-values: {sm_model.pvalues}")
print(f"Confidence Intervals: {sm_model.conf_int()}")

# Create grid for prediction surface
Xvals=np.linspace(min(housing['area']), max(housing['area']),20)
Yvals=np.linspace(min(housing['bedrooms']), max(housing['bedrooms']),20)
Xg, Yg = np.meshgrid(Xvals, Yvals)
Zvals = np.array(multRegMod.intercept_[0] + (Xg * multRegMod.coef_[0,0] +  Yg * multRegMod.coef_[0,1]))

# Plot data and surface
fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.grid()
ax.scatter(housing[['area']], housing[['bedrooms']], housing[['price']], color='#1f77b4')
ax.set_xlabel('Area', fontsize=14)
ax.set_ylabel('Bedrooms', fontsize=14)
ax.set_zlabel('Price ($)', fontsize=14)
ax.plot_surface(Xg, Yg, Zvals, alpha=.25, color='grey')
plt.show()