from flask import Flask, jsonify, request, render_template
import pandas as pd
import requests


app = Flask(__name__)


# ============================================================
# AWS API GATEWAY
# ============================================================

API_URL = "https://l7v2b5890l.execute-api.ap-south-1.amazonaws.com/predict"


# ============================================================
# CUSTOMER DATABASE
# ============================================================

CUSTOMER_FILE = "data/processed/cleaned_telco_churn.csv"

customers = pd.read_csv(CUSTOMER_FILE)


# Create Customer IDs if they don't already exist

if "CustomerID" not in customers.columns:

    customers.insert(
        0,
        "CustomerID",
        [
            "CUST" + str(i).zfill(4)
            for i in range(1, len(customers) + 1)
        ]
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return render_template("index.html")


# ============================================================
# GET CUSTOMER LIST
# ============================================================

@app.route("/customers", methods=["GET"])
def get_customers():

    customer_list = customers[
        ["CustomerID"]
    ].to_dict("records")

    return jsonify(customer_list)


# ============================================================
# GET ONE CUSTOMER
# ============================================================

@app.route("/customer/<customer_id>", methods=["GET"])
def get_customer(customer_id):

    customer = customers[
        customers["CustomerID"] == customer_id
    ]

    if customer.empty:

        return jsonify({
            "error": "Customer not found"
        }), 404


    customer = customer.drop(
        columns=["Churn"],
        errors="ignore"
    )


    return jsonify(
        customer.iloc[0].to_dict()
    )


# ============================================================
# CUSTOMER PREDICTION
# ============================================================

@app.route("/predict_customer", methods=["POST"])
def predict_customer():

    data = request.get_json()


    try:

        response = requests.post(
            API_URL,
            json=data
        )

        response.raise_for_status()

        result = response.json()

        return jsonify(result)


    except requests.exceptions.RequestException as e:

        return jsonify({

            "error": "Unable to connect to prediction service",

            "details": str(e)

        }), 500


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )