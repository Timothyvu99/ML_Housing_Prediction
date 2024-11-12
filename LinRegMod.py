# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm

# Function to evaluate a Linear Regression model
def evaluate_model(data_path, features, target, test_size=0.5, plot_surface=False):
    # Load and prepare the data
    data = pd.read_csv(data_path)
    x = data[features].values
    y = data[[target]].values
    
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    
    # Initialize and fit Linear Regression model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Print intercept and coefficients
    print(f"\nDataset: {data_path}")
    print(f"Intercept: {model.intercept_[0]}")
    print(f"Weights: {model.coef_[0]}")
    
    # Predict on the test set
    y_pred = model.predict(x_test)
    
    # Residual Score on the test set
    residuals = y_test - y_pred

    # Accuracy (R^2 Score) and Adjusted R^2 Score on the test set
    accuracy = model.score(x_test, y_test)
    n = x_test.shape[0]
    p = x_test.shape[1]
    adjusted_accuracy = 1 - (1-accuracy) * (n - 1) / (n - p - 1)
    
    print(f"R^2 Score (Test Set): {accuracy:.2f}")
    print(f"Adjusted R^2 Score (Test Set): {adjusted_accuracy:.2f}")

    # Calculate and print Mean Squared Error and Mean Absolute Error on the test set
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Squared Error (Test Set): {mse:.2f}")
    print(f"Mean Absolute Error (Test Set): {mae:.2f}")
    
    # Statistical summary using statsmodels
    x_train_const = sm.add_constant(x_train)
    sm_model = sm.OLS(y_train, x_train_const).fit()
    print(f"P-values:\n{sm_model.pvalues}")
    print(f"Confidence Intervals:\n{sm_model.conf_int()}")
    
    # Optional 3D plot if only two features are used
    if plot_surface and len(features) == 2:
        X_vals = np.linspace(min(data[features[0]]), max(data[features[0]]), 20)
        Y_vals = np.linspace(min(data[features[1]]), max(data[features[1]]), 20)
        Xg, Yg = np.meshgrid(X_vals, Y_vals)
        Zvals = model.intercept_[0] + (Xg * model.coef_[0][0] + Yg * model.coef_[0][1])
        
        fig = plt.figure(figsize=(12, 6))
        
        # 3D Surface plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(data[features[0]], data[features[1]], data[target], color='#1f77b4')
        ax1.plot_surface(Xg, Yg, Zvals, alpha=0.25, color='grey')
        ax1.set_xlabel(features[0], fontsize=14)
        ax1.set_ylabel(features[1], fontsize=14)
        ax1.set_zlabel(target, fontsize=14)
        ax1.set_title("3D Surface Plot: {data_path}")

        # Residuals Plot
        ax2 = fig.add_subplot(122)
        ax2.plot(y_test, residuals, 'o', markersize=5, color='red')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_xlabel("Observed Values")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residuals Plot")
        
        plt.tight_layout()
        plt.show()

# Evaluate multiple datasets with the function
evaluate_model(
    data_path='Housing.csv',
    features=['area', 'bedrooms'],
    target='price',
    test_size=0.5,
    plot_surface=True
)

evaluate_model(
    data_path='data.csv',
    features=['sqft_living', 'bedrooms'],
    target='price',
    test_size=0.7
)

evaluate_model(
    data_path='house-prices.csv',
    features=['SqFt', 'Bedrooms'],
    target='Price',
    test_size=0.7,
    plot_surface=True
)
