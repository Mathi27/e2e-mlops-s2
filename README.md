# Used Car Price — Inference Service (Flask)

A small REST API + web page that serves the model trained in the MLflow notebook.
Goal of this project: **understand exactly how a client talks to a saved model over HTTP.**

```
car-price-api/
├── app.py                 # Flask app: the routes + model loading
├── config.py              # EASY CONFIG: paths, host, port, debug (env-var overridable)
├── features.py            # raw request  ->  the exact feature table the model expects
├── car_transformers.py    # copied from the training project (src/) - needed to unpickle the model
├── model/
│   ├── best_model.pkl         # <- copy from training project: artifacts/models/
│   └── model_metadata.json    # <- copy from training project: artifacts/models/
├── templates/index.html   # the web page (plain HTML + JS, no libraries)
├── client_example.py      # call the API from Python
└── requirements.txt
```

---

## 1. Setup (5 minutes)

**Step 1 — copy 3 files from the training project into this one**

| From the training project | To this project |
|---|---|
| `artifacts/models/best_model.pkl` | `model/best_model.pkl` |
| `artifacts/models/model_metadata.json` | `model/model_metadata.json` |
| `src/car_transformers.py` | `car_transformers.py` (project root, next to `app.py`) |

**Step 2 — virtual environment + install**

```bash
cd car-price-api
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Step 3 — use the same library versions as training.**
Open `model/model_metadata.json` → `library_versions`, then for example:

```bash
pip install scikit-learn==<version> xgboost==<version>
```

A pickled model is only guaranteed to load with the same versions. `app.py` prints a `[WARN]` at startup, and `/health` reports `version_warnings`, if they differ.

**Step 4 — run**

```bash
python app.py
```

Open **http://127.0.0.1:5001** to see the web page. (Port 5001, because on macOS port 5000 is taken by AirPlay and answers `403`.)
The terminal should print: `[OK] Model loaded: XGBoost (...)`.

### Changing the configuration

Everything is in `config.py`. Edit the file, or override a value without editing:

```bash
PORT=8000 python app.py                         # macOS / Linux
DEBUG=false python app.py                       # no auto-reload / debugger
MODEL_PATH=/other/best_model.pkl python app.py  # use another model file
HOST=0.0.0.0 python app.py                      # reachable from other computers on your network
# Windows PowerShell:  $env:PORT="8000"; python app.py
```

---

## 2. How the client talks to the saved model

```
 ┌────────────┐   1. HTTP POST /predict             ┌──────────────────────────────────────────┐
 │  Browser   │      Content-Type: application/json │  Flask  (app.py)                         │
 │  curl      │ ──────────────────────────────────► │                                          │
 │  Python    │      {"brand":"BMW","model_year":..}│  a. parse JSON       request.get_json()  │
 │  mobile    │                                     │  b. validate         features.validate   │
 │  app ...   │                                     │  c. build features   features.build_...  │
 │            │   4. HTTP 200 + JSON                │  d. model.predict(features) <- best_model.pkl
 │            │ ◄────────────────────────────────── │     (loaded ONCE at startup)             │
 └────────────┘      {"predicted_price": 23450, ...}│  e. jsonify(result)                      │
                                                    └──────────────────────────────────────────┘
```

The model never "listens" on the network. It is just a Python object in memory.
**Flask is the translator:** HTTP + JSON on the outside, a pandas DataFrame and `model.predict()` on the inside.

### The 5 steps inside `/predict` (open `app.py` and follow along)

1. **Load once.** `joblib.load("model/best_model.pkl")` runs when the server starts and the model stays in memory (`STATE["model"]`). Loading it on every request would be very slow.
2. **Parse.** `request.get_json()` turns the request body into a Python dict.
3. **Validate** (`features.validate`). Missing field? Wrong type? Unknown fuel type? Then HTTP `400` with a clear message. Never trust client input.
4. **Build features** (`features.build_features`). The model was trained on columns such as `Vehicle_Age`, `Mileage_per_Year`, `Age_Old` and `fuel_type` (as an integer). A client should not need to know that. So the API accepts friendly fields (`model_year`, `mileage`, `"Gasoline"`) and converts them *exactly like the training notebook did*, using the definitions saved in `model_metadata.json`:

   | Client sends | Feature the model receives | How |
   |---|---|---|
   | `model_year: 2018` | `Vehicle_Age = 7` | `2025 − 2018` (reference year from metadata) |
   | `mileage: 45000` | `Mileage_per_Year = 6428.6` | `mileage / age` |
   | `mileage: 45000` | `Milage_High = 1`, ... | quartile bin edges saved in metadata |
   | `fuel_type: "Gasoline"` | `fuel_type = 2` | position in `label_encoder_classes` from metadata |
   | `brand: "BMW"` | `brand` stays text | encoded **inside** the model (target encoding learned on training data) |
   | `accident: false` | `Accident_Impact = 0` | bool to 0/1 |

5. **Predict.** `model.predict(features)` runs the full saved pipeline: brand encoder → scaler → estimator → undo the log-transform → **price in dollars**. Flask wraps it in JSON.

> Rule of thumb for teaching: **whatever happened to the data before `fit()` must happen again before `predict()`.**
> Part of that is inside the pickle (encoder, scaler, log transform). The rest lives in `features.py`.

---

## 3. The REST API

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health` | Is the server up and the model loaded? |
| `POST` | `/predict` | Send a car, get a price |
| `GET`  | `/model-info` | Model name, metrics, allowed brands / fuel types / transmissions (the web page uses it to fill its dropdowns) |
| `GET`  | `/` | The web page |

### `GET /health`

```bash
curl http://127.0.0.1:5001/health
```

```json
{"status":"ok","model_loaded":true,"model_name":"XGBoost",
 "model_trained_at":"2026-10-06T00:20:38+00:00","version_warnings":[]}
```

Returns **200** when ready, **503** if the model failed to load (the body says why). Docker, Kubernetes and load balancers call an endpoint like this to decide whether to send traffic.

### `POST /predict`

Request body (all 10 fields required):

| Field | Type | Example | Rule |
|---|---|---|---|
| `brand` | text | `"BMW"` | any text; unknown brands work but return a warning |
| `model_year` | integer | `2018` | 1980 to 2025 |
| `mileage` | number | `45000` | 0 to 1,000,000 (miles) |
| `fuel_type` | text | `"Gasoline"` | one of the values listed by `/model-info` (case-insensitive) |
| `transmission` | text | `"A/T"` | one of the values listed by `/model-info` (`A/T`, `M/T`, `CVT`, `OTHER`) |
| `hp` | number | `300` | 20 to 1500 |
| `engine_displacement` | number | `3.0` | litres, 0.5 to 10 |
| `is_v_engine` | boolean | `true` | |
| `accident` | boolean | `false` | accident or damage reported |
| `clean_title` | boolean | `true` | |

```bash
curl -X POST http://127.0.0.1:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"brand":"BMW","model_year":2018,"mileage":45000,"fuel_type":"Gasoline","transmission":"A/T","hp":300,"engine_displacement":3.0,"is_v_engine":true,"accident":false,"clean_title":true}'
```

```json
{"predicted_price": 28928, "currency": "USD", "model": "XGBoost", "warnings": [], "inference_ms": 9.9}
```

**Status codes** — a good REST API tells the client *what kind* of problem happened:

| Code | Meaning | Example |
|---|---|---|
| `200` | success | prediction returned |
| `400` | the client sent bad data | `{"error":"Invalid input","details":["'fuel_type' must be one of [...]"]}` |
| `415` | body is not JSON / header missing | forgot `Content-Type: application/json` |
| `503` | server is up but the model is not loaded | wrong `MODEL_PATH` |
| `500` | unexpected server bug | details are logged on the server |

### From Python (`client_example.py`)

```python
import requests
r = requests.post("http://127.0.0.1:5001/predict", json={...})   # json= sets the header and serialises the dict
print(r.status_code, r.json())
```

Start the server in one terminal, then run `python client_example.py` in another.

---

## 4. The web page (`templates/index.html`)

Plain HTML plus a few lines of JavaScript. It is **just another client** of the API: it has no model and no Python.

* On load it calls `fetch("/health")` and `fetch("/model-info")` to fill the status badge and the dropdowns.
* On submit it builds the JSON from the form and calls `fetch("/predict", {method:"POST", ...})`.
* It shows the **request JSON**, the **response JSON** and the equivalent **curl command**, so students see that the page and `curl` send the same thing.

Classroom tip: open the browser **Developer Tools → Network** tab, press *Predict price*, and click the `predict` request to see headers, payload and response.

---

## 5. Teaching exercises

1. Call `/health` with the model file renamed: see the `503` and the message. Rename it back and restart.
2. Send `{"brand":"BMW"}`: read the `400` message. Add fields until it succeeds.
3. Same car, change only `model_year` from 2024 to 2010: watch the price move.
4. Change `brand` to `"Zzz"`: success with a **warning** (what does the model do for unseen brands?).
5. Send the request without `-H "Content-Type: application/json"`: you get `415`. Why does the server care?
6. Change `PORT` in `config.py` and call the new port.
7. Add a new endpoint `GET /version` that returns the library versions (hint: `STATE["meta"]["library_versions"]`).

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: car_transformers` | copy `src/car_transformers.py` from the training project next to `app.py` |
| `[ERROR] Could not load model ... FileNotFoundError` | copy `best_model.pkl` + `model_metadata.json` into `model/` (or fix `MODEL_PATH`) |
| `[WARN] scikit-learn: trained with X but installed Y` | `pip install scikit-learn==X` (same for xgboost) |
| Browser shows `403` on port 5000 | macOS AirPlay uses 5000; this project uses 5001 |
| `Address already in use` | another program uses the port: `PORT=5050 python app.py` |
| Page loads but dropdowns are empty | model not loaded: open `/health` to see why |

## 7. Notes for later

* `python app.py` uses Flask's **development server** (fine for teaching). For real deployments run it behind a production server, for example `pip install gunicorn` then `gunicorn -w 2 -b 0.0.0.0:5001 app:app` (Linux/macOS). That is also what you would put in a Docker image.
* A `.pkl` file is **trusted code**: never load one from a source you don't trust.
* This API predicts one car per request. Batch prediction, authentication and request logging are natural next steps.
