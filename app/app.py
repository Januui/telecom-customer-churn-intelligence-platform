from flask import Flask, jsonify, request
import joblib
import pandas as pd

# Create the Flask application
app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load("models/tuned_random_forest.pkl")


# Home endpoint

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Telecom Customer Churn Prediction API is running"
    })
# Prediction Endpoint

@app.route("/predict", methods=["POST"])
def predict():
    # Get customer data from the request
    data = request.get_json()

    # Convert the input data into a DataFrame
    customer_data = pd.DataFrame([data])

    # Arrange columns in the same exact order used during model training.
    customer_data = customer_data[model.feature_names_in_]

    # Generate churn prediction
    prediction = model.predict(customer_data)

    # Generate churn probabilities
    probability = model.predict_proba(customer_data)

    # Return prediction and probabilities
    return jsonify({
        "Churn_prediction": int(prediction[0]),
        "Probability_no_churn": float(probability[0][0]),
        "Probability_churn": float(probability[0][1])
    })

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)