# Machine Learning Housing Price Prediction Project
## Course
### CPSC: 483 - Introduction to Machine Learning

## Description
This application was created to find the best Machine Learning Model to predict the housing prices in California. Utilizing datasets from Kaggle, we've
calculated the accuracy score from each model to determine which would best fit in this application.

## Contributors
Bryan Rivas \
Timothy Vu \
Justin Dong

## Research Findings
### Model Performance Metrics (Ordered by R² x 100)

| Model                    | R² x 100 (%) | Mean Squared Error (MSE)     | Root Mean Squared Error (RMSE) | Mean Absolute Error (MAE)  |
|--------------------------|--------------|------------------------------|--------------------------------|----------------------------|
| Polynomial Regression    | 68.48        | 1,463,188,621,192.95         | $1,209,623.34                 | $894,967.45               |
| Linear Regression        | 67.55        | 1,506,230,725,917.46         | $1,227,285.92                 | $902,975.64               |
| Random Forest Regressor  | 66.31        | 1,563,670,698,302.80         | $1,250,468.19                 | $902,726.41               |
| Neural Network           | 62.89        | 1,875,913,192,473.16         | $1,369,639.80                 | $1,006,581.19             |
| Naive Bayes              | 57.65        | 0.28                         | $0.53                         | $0.17                     |
| Decision Tree            | 42.79        | 2,463,688,650,437.66         | $1,569,614.17                 | $1,186,486.17             |
| Support Vector Regression| -3.62        | 4,462,186,287,255.04         | $2,112,388.76                 | $1,578,104.56             |
