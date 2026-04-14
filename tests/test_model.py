"""Unit tests for the ML project."""

import numpy as np
import pytest

from ml_project.model import load_data, train_model, evaluate_model, predict_sample


@pytest.fixture
def dataset():
    """Provide train/test split for tests."""
    return load_data()


@pytest.fixture
def trained_model(dataset):
    """Provide a trained model for tests."""
    x_train, _, y_train, _, _ = dataset
    return train_model(x_train, y_train)


def test_load_data_shapes(dataset):
    """Check that data is split correctly."""
    x_train, x_test, y_train, y_test, names = dataset

    assert len(x_train) > 0
    assert len(x_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
    assert len(names) == 3


def test_train_model_returns_fitted_model(dataset):
    """Check model can be trained."""
    x_train, _, y_train, _, _ = dataset
    model = train_model(x_train, y_train)

    assert hasattr(model, "predict")


def test_evaluate_model_accuracy(dataset):
    """Check model quality is reasonable."""
    x_train, x_test, y_train, y_test, _ = dataset
    model = train_model(x_train, y_train)
    accuracy = evaluate_model(model, x_test, y_test)

    assert accuracy > 0.8


def test_predict_sample(trained_model):
    """Check a prediction is returned."""
    sample = np.array([5.1, 3.5, 1.4, 0.2])
    pred = predict_sample(trained_model, sample)

    assert pred in [0, 1, 2]