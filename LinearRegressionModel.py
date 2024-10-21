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