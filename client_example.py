"""
client_example.py - talk to the API from Python (no browser). Start the server first:  python app.py
This is exactly what any other program (a mobile app, another service, a notebook...) would do.
"""
import requests

BASE_URL = "http://127.0.0.1:5001"      # must match HOST/PORT in config.py

# 1) health check
r = requests.get(f"{BASE_URL}/health", timeout=5)
print("GET /health   ->", r.status_code, r.json())

# 2) a valid prediction
car = {
    "brand": "BMW", "model_year": 2018, "mileage": 45000,
    "fuel_type": "Gasoline", "transmission": "A/T",
    "hp": 300, "engine_displacement": 3.0,
    "is_v_engine": True, "accident": False, "clean_title": True,
}
r = requests.post(f"{BASE_URL}/predict", json=car, timeout=10)        # json= sets the Content-Type header for us
print("POST /predict ->", r.status_code, r.json())

# 3) an invalid request: see how the API reports mistakes (HTTP 400)
bad = {**car, "model_year": "last year", "fuel_type": "Water"}
r = requests.post(f"{BASE_URL}/predict", json=bad, timeout=10)
print("POST /predict (bad input) ->", r.status_code, r.json())
