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

        if "features" not in data:
            return jsonify({"error": "Missing 'features' key"}), 400

        features = np.array(data["features"])

        if features.shape[0] != 4:
            return jsonify({"error": "Expected 4 features"}), 400

        prediction = model.predict([features])[0]

        return jsonify({"prediction": int(prediction)})

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
