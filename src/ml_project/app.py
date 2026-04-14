"""Flask API for Iris prediction."""

from pathlib import Path
import joblib
import numpy as np
from flask import Flask, request, jsonify

MODEL_PATH = Path("artifacts/iris_model.joblib")

app = Flask(__name__)

# Load model once at startup
model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    """Health check endpoint."""
    return {"message": "Iris ML API is running"}


@app.route("/predict", methods=["POST"])
def predict():
    """Predict Iris class from input features."""
    try:
        data = request.get_json()

        # Expect input like: {"features": [5.1, 3.5, 1.4, 0.2]}
        features = np.array(data["features"])

        prediction = model.predict([features])[0]

        return jsonify({
            "prediction": int(prediction)
        })

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)