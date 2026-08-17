import os
import json
import joblib
import pandas as pd


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

        return pd.DataFrame(data)

    raise ValueError(
        f"Unsupported content type: {request_content_type}"
    )


def predict_fn(input_data, model):
    """
    Generate prediction and probabilities.
    """

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