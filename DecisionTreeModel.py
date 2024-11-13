# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from scipy.stats import randint, uniform
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings("ignore", category=UserWarning, message="Found unknown categories")
# Load the dataset
data = pd.read_csv("Housing.csv")

# Define features and target variable
X = data.drop(columns=['price'])
y = data['price']

# Enhanced Preprocessing: One-hot encode, add polynomial and interaction terms, and scale
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['street', 'city']),
        ('num', Pipeline([
            ('scaler', StandardScaler()),  # Scaling before polynomial features
            ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),
            ('pca', PCA(n_components=10))  # Reduces complexity if there are many polynomial features
        ]), X.select_dtypes(include='number').columns)
    ],
    remainder='passthrough'
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define a Random Forest model with expanded hyperparameter tuning using RandomizedSearchCV
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Set up Randomized Search for Random Forest hyperparameters
rf_param_dist = {
    'regressor__n_estimators': randint(100, 500),
    'regressor__max_depth': [None, 10, 20, 30, 40],
    'regressor__min_samples_split': randint(2, 15),
    'regressor__min_samples_leaf': randint(1, 10),
    'regressor__max_features': ['sqrt', 'log2', None]
}
rf_random_search = RandomizedSearchCV(
    rf_pipeline, rf_param_dist, n_iter=50, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42
)

# Train the Random Forest model with Randomized Search
rf_random_search.fit(X_train, y_train)
y_pred_rf = rf_random_search.predict(X_test)

# Evaluate Random Forest performance
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)
print("Random Forest Best Params:", rf_random_search.best_params_)
print("Random Forest Regressor:")
print(f"Mean Squared Error: {rf_mse:.2f}")
print(f"Mean Absolute Error: {rf_mae:.2f}")
print(f"R^2 Score: {rf_r2:.2f}")

# Define Gradient Boosting model with expanded hyperparameter tuning using RandomizedSearchCV
gb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# Set up Randomized Search for Gradient Boosting hyperparameters
gb_param_dist = {
    'regressor__n_estimators': randint(100, 400),
    'regressor__learning_rate': uniform(0.01, 0.2),
    'regressor__max_depth': randint(3, 10),
    'regressor__subsample': uniform(0.5, 0.5)
}
gb_random_search = RandomizedSearchCV(
    gb_pipeline, gb_param_dist, n_iter=50, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42
)

# Train the Gradient Boosting model with Randomized Search
gb_random_search.fit(X_train, y_train)
y_pred_gb = gb_random_search.predict(X_test)

# Evaluate Gradient Boosting performance
gb_mse = mean_squared_error(y_test, y_pred_gb)
gb_mae = mean_absolute_error(y_test, y_pred_gb)
gb_r2 = r2_score(y_test, y_pred_gb)
print("Gradient Boosting Best Params:", gb_random_search.best_params_)
print("Gradient Boosting Regressor:")
print(f"Mean Squared Error: {gb_mse:.2f}")
print(f"Mean Absolute Error: {gb_mae:.2f}")
print(f"R^2 Score: {gb_r2:.2f}")

# Residual Analysis for Random Forest (optional, helps in further tuning)
rf_residuals = y_test - y_pred_rf
print("Residual Analysis for Random Forest")
print("Mean Residuals:", rf_residuals.mean())
print("Std of Residuals:", rf_residuals.std())

# Optional: Feature importance from Random Forest
best_rf = rf_random_search.best_estimator_['regressor']
if hasattr(best_rf, "feature_importances_"):
    importances = best_rf.feature_importances_
    feature_names = rf_pipeline.named_steps['preprocessor'].get_feature_names_out()
    feature_importances = pd.DataFrame(importances, index=feature_names, columns=["importance"]).sort_values(by="importance", ascending=False)
    print("Top 10 Important Features:")
    print(feature_importances.head(10))
