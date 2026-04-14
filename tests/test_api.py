import json
import pytest
from ml_project.api import app


@pytest.fixture
def client():
    """Flask test client."""
    app.config["TESTING"] = True
    return app.test_client()


def test_home(client):
    """Test health endpoint."""
    response = client.get("/")
    assert response.status_code == 200


def test_predict(client):
    """Test prediction endpoint."""
    response = client.post(
        "/predict",
        data=json.dumps({"features": [5.1, 3.5, 1.4, 0.2]}),
        content_type="application/json",
    )

    data = response.get_json()
    assert response.status_code == 200
    assert "prediction" in data