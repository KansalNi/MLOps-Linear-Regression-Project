from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("linear_regression_model.joblib")

app = Flask(__name__)

@app.route('/')
def home():
    return "Hi, Welcome to the Housing Price Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json(force=True)
        
        # Convert data into a numpy array
        features = np.array(data['features'])
        
        # Ensure the input is 2D (for a single sample, reshape to (1, -1))
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Use the model to predict the price
        prediction = model.predict(features)
        
        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
