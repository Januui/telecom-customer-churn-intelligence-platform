from flask import Flask, jsonify

# Create the Flask application
app = Flask(__name__)


# Home endpoint
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Telecom Customer Churn Prediction API is running"
    })


# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)