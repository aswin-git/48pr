"""Train and save the Iris model."""

from pathlib import Path
import joblib

from ml_project.model import load_data, train_model, evaluate_model


MODEL_PATH = Path("artifacts/iris_model.joblib")


def main():
    """Train model and save it to disk."""
    x_train, x_test, y_train, y_test, _ = load_data()
    model = train_model(x_train, y_train)
    accuracy = evaluate_model(model, x_test, y_test)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Test accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()


