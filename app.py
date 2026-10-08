"""
app.py - a tiny Flask REST API around the saved used-car price model.

    GET  /            -> web page (a form that calls the API)
    GET  /health      -> is the service up and is the model loaded?
    GET  /model-info  -> what the model expects (used by the web page to fill the dropdowns)
    POST /predict     -> JSON in, predicted price (JSON) out

Run:  python app.py      then open http://127.0.0.1:5001
"""
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone

import joblib
from flask import Flask, Response, g, jsonify, render_template, request

import config
import monitoring
from features import ValidationError, build_features, validate

# The pickled model refers to the class `car_transformers.BrandTargetEncoder`.
# Python must be able to import that module BEFORE joblib.load() can rebuild the model.
sys.path.insert(0, str(config.BASE_DIR))
import car_transformers  # noqa: E402,F401

app = Flask(__name__)

logger = monitoring.setup_logging(config.LOG_LEVEL)     # JSON logs on stdout
metrics = monitoring.Metrics()                           # in-memory counters, shown at /metrics
QUIET_ROUTES = {"/health", "/metrics"}                   # called every few seconds by Docker/Prometheus: log at DEBUG only

# ----------------------------------------------------------------------------
# 1) Load the model ONCE when the server starts (not on every request - that would be slow)
# ----------------------------------------------------------------------------
STATE = {"model": None, "meta": None, "error": None, "loaded_at": None, "warnings": []}


def load_model():
    try:
        STATE["model"] = joblib.load(config.MODEL_PATH)
        STATE["meta"] = json.loads(config.METADATA_PATH.read_text())
        STATE["loaded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        STATE["error"] = None
        STATE["warnings"] = version_warnings(STATE["meta"])
        print(f"[OK] Model loaded: {STATE['meta']['model_name']}  ({config.MODEL_PATH})")
        for w in STATE["warnings"]:
            print("[WARN]", w)
    except Exception as exc:                      # the server still starts; /health will say what is wrong
        STATE["model"], STATE["meta"], STATE["error"] = None, None, f"{type(exc).__name__}: {exc}"
        print("[ERROR] Could not load model ->", STATE["error"])


def version_warnings(meta):
    """Pickles are fragile across library versions - warn if they differ from the training environment."""
    import sklearn
    import xgboost
    trained = meta.get("library_versions", {})
    now = {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}
    return [f"{lib}: trained with {trained[lib]} but installed {ver}"
            for lib, ver in now.items() if lib in trained and trained[lib] != ver]


def known_brands():
    return list(STATE["model"].regressor_.named_steps["brand_encoder"].mapping_.index)


load_model()


# ----------------------------------------------------------------------------
# Request / response logging + metrics (runs around EVERY request)
# ----------------------------------------------------------------------------
@app.before_request
def start_request():
    g.start = time.perf_counter()
    g.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    g.log_fields = {}                                    # routes add extra (safe!) details here


@app.after_request
def finish_request(response):
    seconds = time.perf_counter() - g.start
    route = request.url_rule.rule if request.url_rule else "unmatched"     # route pattern, not raw URL -> bounded metric labels
    metrics.observe_request(request.method, route, response.status_code, seconds)
    response.headers["X-Request-ID"] = g.request_id                         # lets a client quote the id when reporting a problem

    level = logging.DEBUG if route in QUIET_ROUTES else (
        logging.ERROR if response.status_code >= 500 else logging.WARNING if response.status_code >= 400 else logging.INFO)
    logger.log(level, "request", extra={"fields": {
        "request_id": g.request_id, "method": request.method, "route": route,
        "status": response.status_code, "latency_ms": round(seconds * 1000, 1), **g.log_fields}})
    return response


# ----------------------------------------------------------------------------
# 2) Endpoints
# ----------------------------------------------------------------------------
@app.get("/")
def home():
    return render_template("index.html", title=config.APP_TITLE)


@app.get("/health")
def health():
    """Health check: used by humans, load balancers, Docker HEALTHCHECK, Kubernetes probes..."""
    if STATE["model"] is None:
        return jsonify(status="error", model_loaded=False, detail=STATE["error"]), 503
    return jsonify(status="ok", model_loaded=True,
                   model_name=STATE["meta"]["model_name"],
                   model_trained_at=STATE["meta"]["trained_at_utc"],
                   server_loaded_model_at=STATE["loaded_at"],
                   version_warnings=STATE["warnings"])


@app.get("/metrics")
def metrics_endpoint():
    """Counters and latency histogram in Prometheus text format (readable by humans too)."""
    return Response(metrics.render(), mimetype="text/plain; version=0.0.4")


@app.get("/model-info")
def model_info():
    if STATE["model"] is None:
        return jsonify(error="Model not loaded", detail=STATE["error"]), 503
    meta = STATE["meta"]
    classes = meta["feature_preparation"]["label_encoder_classes"]
    metrics = meta["metrics"]
    return jsonify(
        model_name=meta["model_name"],
        brands=sorted(known_brands()),
        fuel_types=classes["fuel_type"],
        transmissions=classes["transmission"],
        required_fields=["brand", "model_year", "mileage", "fuel_type", "transmission",
                         "hp", "engine_displacement", "is_v_engine", "accident", "clean_title"],
        test_metrics={k: round(metrics[k], 3) for k in ("test_rmse", "test_mae", "test_r2", "test_mape")},
        reference_year=meta["feature_preparation"].get("vehicle_age_reference_year", 2025),
    )


@app.post("/predict")
def predict():
    """
    Request  : JSON body, e.g. {"brand": "BMW", "model_year": 2018, "mileage": 45000, ...}
    Response : {"predicted_price": 23450, "currency": "USD", ...}
    Errors   : 400 invalid input | 415 not JSON | 503 model not loaded | 500 unexpected
    """
    if STATE["model"] is None:
        return jsonify(error="Model not loaded", detail=STATE["error"]), 503

    payload = request.get_json(silent=True)           # None if the body is not valid JSON
    if payload is None:
        return jsonify(error="Send a JSON body with header 'Content-Type: application/json'"), 415

    if config.LOG_PAYLOADS:
        g.log_fields["input"] = monitoring.safe_payload(payload)            # allow-list + redaction, never the raw body

    started = time.perf_counter()
    try:
        clean, warnings = validate(payload, STATE["meta"], known_brands())   # 1. check the input
        features = build_features(clean, STATE["meta"])                      # 2. raw input -> model features
        price = float(STATE["model"].predict(features)[0])                   # 3. the actual prediction
    except ValidationError as exc:
        g.log_fields["validation_errors"] = exc.errors
        return jsonify(error="Invalid input", details=exc.errors), 400
    except Exception as exc:                          # never leak a stack trace to the client
        logger.exception("prediction failed", extra={"fields": {"request_id": g.request_id}})
        return jsonify(error="Prediction failed", detail=str(exc)), 500

    g.log_fields.update(predicted_price=round(price), model=STATE["meta"]["model_name"], warnings=len(warnings))
    metrics.count_prediction(STATE["meta"]["model_name"])
    return jsonify(predicted_price=round(price), currency="USD",
                   model=STATE["meta"]["model_name"], warnings=warnings,
                   inference_ms=round((time.perf_counter() - started) * 1000, 1))


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
