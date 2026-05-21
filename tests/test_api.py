from app.app import app


def test_health():
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200


def test_predict():
    client = app.test_client()
    response = client.post(
        "/predict",
        json={"features": [0] * 30}
    )

    assert response.status_code == 200
    assert "prediction" in response.get_json()