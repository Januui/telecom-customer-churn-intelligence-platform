import os
import json
import joblib
import pandas as pd
import numpy as np


def model_fn(model_dir):
    """
    Load the trained Random Forest model.
    """

    model_path = os.path.join(
        model_dir,
        "tuned_random_forest.pkl"
    )

    model = joblib.load(model_path)

    return model


def input_fn(request_body, request_content_type):
    """
    Convert incoming JSON into a DataFrame.
    """

    if request_content_type == "application/json":

        data = json.loads(request_body)

        return pd.DataFrame([data])

    raise ValueError(
        f"Unsupported content type: {request_content_type}"
    )


def preprocess_customer_data(customer, model):
    """
    Convert 19 customer inputs into the 49 features
    expected by the trained Random Forest model.
    """

    # 1. Create TotalServices

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


    # 2. Create LongTermCustomer

    customer["LongTermCustomer"] = np.where(
        customer["tenure"] >= 24,
        1,
        0
    )


    # 3. Create TenureGroup

    customer["TenureGroup"] = pd.cut(
        customer["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"]
    )


    # 4. Create MonthlyChargeCategory

    customer["MonthlyChargeCategory"] = pd.cut(
        customer["MonthlyCharges"],
        bins=[0, 35, 70, 120],
        labels=["Low", "Medium", "High"]
    )


    # 5. Convert binary columns

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


    # 6. One-hot encode categorical columns

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


    # 7. Make sure all 49 model features exist

    for column in model.feature_names_in_:

        if column not in customer.columns:
            customer[column] = 0


    # 8. Keep only the 49 model features
    #    and arrange them in the exact model order

    customer = customer[model.feature_names_in_]


    return customer


def predict_fn(input_data, model):
    """
    Convert 19 customer inputs into 49 model features
    and generate prediction and probabilities.
    """

    # Convert 19 inputs → 49 model features

    input_data = preprocess_customer_data(
        input_data,
        model
    )


    # Make prediction

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)


    return {
        "Churn_prediction": int(prediction[0]),
        "Probability_no_churn": float(probability[0][0]),
        "Probability_churn": float(probability[0][1])
    }


def output_fn(prediction, accept):
    """
    Convert prediction into JSON response.
    """

    if accept == "application/json":

        return json.dumps(prediction), "application/json"

    raise ValueError(
        f"Unsupported accept type: {accept}"
    )