# Import packages and functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression

#Load the dataset
housing_all = pd.read_csv('Housing.csv')

#Keep subset of features, and drop missing values
housing = housing_all[['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']]

#Define input and output features
x = housing[['area', 'bedrooms']].values.reshape(-1, 2)
y = housing[['price']].values.reshape(-1, 1)

#initialize multiple linear regression model
multRegMod = LinearRegression()

#Fit multiple linear regression model
multRegMod.fit(x, y)

#Print intercept and weight
print(f"Multiple Regression Model Intercept: {multRegMod.intercept_}")
print(f"Multiple Regression Model Weight: {multRegMod.coef_}")

yPredicted = multRegMod.predict(x)

print(f"Predicted Price: {yPredicted[:6]}")

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