# MLOps-Linear-Regression-Project
MLOps Assignment - Linear Regression on the California Housing Dataset


train.py

Implementing simple linear regression on the California Housing Dataset, 
- fetching the dataset, 
- splitting it into training and test sets, 
- fitting a linear regression model, 
- saving the model to a linear_regression_model.joblib file.



app.py

Flask application that uses the trained linear regression model (linear_regression_model.joblib) to predict housing prices.
With a /predict endpoint that accepts POST requests and uses the trained model to predict housing prices.

Run the Flask application:

python app.py


Testing the API:

Command for the welcome message:
curl -X GET http://127.0.0.1:5000/ -H "Content-Type: application/json"

Output:
Welcome to the Housing Price Prediction API!


Command for predicting prices:
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"features": [8.3252, 41.0, 6.984127, 1.02381, 322.0, 2.555556, 37.88, -122.23]}'
