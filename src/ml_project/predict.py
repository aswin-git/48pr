"""Load a saved model and make predictions."""

from pathlib import Path
import joblib

MODEL_PATH = Path("artifacts/iris_model.joblib")


def predict(sample):
    """Return prediction from saved model."""
    model = joblib.load(MODEL_PATH)
    return model.predict([sample])[0]


if __name__ == "__main__":
    example = [5.1, 3.5, 1.4, 0.2]
    print("Predicted class:", predict(example))
