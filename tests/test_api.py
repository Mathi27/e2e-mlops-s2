import pytest
from app import app

VALID_CAR = {
    "brand": "BMW", "model_year": 2018, "mileage": 45000,
    "fuel_type": "Gasoline", "transmission": "A/T",
    "hp": 300, "engine_displacement": 3.0,
    "is_v_engine": True, "accident": False, "clean_title": True,
}


@pytest.fixture
def client():
    return app.test_client()          # Flask calls the app in-process, no server needed


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["model_loaded"] is True


def test_predict_returns_positive_price(client):
    r = client.post("/predict", json=VALID_CAR)
    assert r.status_code == 200
    assert r.get_json()["predicted_price"] > 0


def test_predict_rejects_missing_fields(client):
    assert client.post("/predict", json={"brand": "BMW"}).status_code == 400


def test_predict_rejects_bad_values(client):
    assert client.post("/predict", json={**VALID_CAR, "fuel_type": "Water"}).status_code == 400


def test_predict_rejects_non_json(client):
    assert client.post("/predict", data="hello").status_code == 415


def test_older_car_is_cheaper(client):
    new = client.post("/predict", json={**VALID_CAR, "model_year": 2024}).get_json()["predicted_price"]
    old = client.post("/predict", json={**VALID_CAR, "model_year": 2008}).get_json()["predicted_price"]
    assert old < new                  # a sanity check on the model itself