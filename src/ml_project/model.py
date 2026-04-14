"""Core ML utilities for the Iris classifier."""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def load_data(test_size=0.2, random_state=42):
    """Load Iris data and split into train and test sets."""
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=test_size,
        random_state=random_state,
        stratify=iris.target,
    )
    return x_train, x_test, y_train, y_test, iris.target_names


def train_model(x_train, y_train):
    """Train a simple classifier."""
    model = LogisticRegression(max_iter=200)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """Return model accuracy."""
    return model.score(x_test, y_test)


def predict_sample(model, sample):
    """Predict class index for one sample."""
    return model.predict([sample])[0]
