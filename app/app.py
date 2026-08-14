from flask import Flask, jsonify, request, render_template
import joblib
import pandas as pd
import numpy as np

# Create the Flask application
app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load("models/tuned_random_forest.pkl")


# HOME API

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Telecom Customer Churn Prediction API is running"
    })



# EXISTING 49-FEATURE PREDICTION ENDPOINT


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    customer_data = pd.DataFrame([data])

    # Arrange columns in exact model order
    customer_data = customer_data[model.feature_names_in_]

    prediction = model.predict(customer_data)

    probability = model.predict_proba(customer_data)

    return jsonify({
        "Churn_prediction": int(prediction[0]),
        "Probability_no_churn": float(probability[0][0]),
        "Probability_churn": float(probability[0][1])
    })



# USER-FRIENDLY CUSTOMER PREDICTION ENDPOINT


@app.route("/predict_customer", methods=["POST"])
def predict_customer():

    data = request.get_json()


    # 1. Create DataFrame from normal customer information
    

    customer = pd.DataFrame([data])

   
    # 2. Create TotalServices

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    service_data = customer[service_columns].replace({
        "Yes": 1,
        "No": 0,
        "No internet service": 0,
        "No phone service": 0,
        "DSL": 1,
        "Fiber optic": 1
    })

    customer["TotalServices"] = service_data.sum(axis=1)

    
    # 3. Create LongTermCustomer
   

    customer["LongTermCustomer"] = np.where(
        customer["tenure"] >= 24,
        1,
        0
    )

    
    # 4. Create TenureGroup
    

    customer["TenureGroup"] = pd.cut(
        customer["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"]
    )

    # 5. Create MonthlyChargeCategory
   

    customer["MonthlyChargeCategory"] = pd.cut(
        customer["MonthlyCharges"],
        bins=[0, 35, 70, 120],
        labels=["Low", "Medium", "High"]
    )

    
    # 6. Convert binary columns
   

    binary_columns = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling"
    ]

    for col in binary_columns:

        customer[col] = customer[col].map({
            "Yes": 1,
            "No": 0,
            "Male": 1,
            "Female": 0
        })

    # SeniorCitizen is already expected as 0 or 1
    customer["SeniorCitizen"] = customer["SeniorCitizen"].astype(int)

    
    # 7. One-hot encode categorical columns

    multi_columns = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
        "TenureGroup",
        "MonthlyChargeCategory"
    ]

    customer = pd.get_dummies(
        customer,
        columns=multi_columns,
        dtype=int
    )

    # 8. Make sure all 49 model features exist
  
    for column in model.feature_names_in_:

        if column not in customer.columns:
            customer[column] = 0


    # 9. Keep ONLY the 49 model features
    #    and arrange them in the correct order
    
    customer = customer[model.feature_names_in_]

    # 10. Make prediction
    
    prediction = model.predict(customer)

    probability = model.predict_proba(customer)

    # 11. Return result

    return jsonify({
        "Churn_prediction": int(prediction[0]),
        "Probability_no_churn": float(probability[0][0]),
        "Probability_churn": float(probability[0][1])
    })

# RUN FLASK
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )