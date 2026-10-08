import io
import json
import logging

import pytest

from app import app, logger
from monitoring import JsonFormatter

VALID_CAR = {
    "brand": "BMW", "model_year": 2018, "mileage": 45000,
    "fuel_type": "Gasoline", "transmission": "A/T",
    "hp": 300, "engine_displacement": 3.0,
    "is_v_engine": True, "accident": False, "clean_title": True,
}


@pytest.fixture
def client():
    return app.test_client()


@pytest.fixture
def captured_logs():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    yield stream
    logger.removeHandler(handler)


def test_metrics_count_requests_and_predictions(client):
    client.post("/predict", json=VALID_CAR)
    text = client.get("/metrics").get_data(as_text=True)
    assert 'http_requests_total{method="POST",route="/predict",status="200"}' in text
    assert "http_request_duration_seconds_count" in text
    assert "predictions_total" in text


def test_request_is_logged_as_json_with_latency(client, captured_logs):
    client.post("/predict", json=VALID_CAR)
    entry = json.loads(captured_logs.getvalue().strip().splitlines()[-1])
    assert entry["route"] == "/predict" and entry["status"] == 200
    assert entry["latency_ms"] >= 0 and "predicted_price" in entry


def test_sensitive_values_never_reach_the_logs(client, captured_logs):
    client.post("/predict", json={**VALID_CAR, "email": "secret@example.com", "password": "hunter2", "note": "leak-me"})
    out = captured_logs.getvalue()
    assert "secret@example.com" not in out
    assert "hunter2" not in out
    assert "leak-me" not in out                      # unknown fields are dropped
    assert "[REDACTED]" in out


def test_request_id_header_returned(client):
    r = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"
