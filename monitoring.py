"""
monitoring.py - structured (JSON) logging + simple in-app metrics.

Three ideas:
  1. LOGS    : one JSON line per request (who called what, status, how long) -> stdout -> `docker compose logs`
  2. PRIVACY : the request body is never logged as-is. Only an allow-list of fields is kept; sensitive keys are redacted.
  3. METRICS : counters + a latency histogram kept in memory and shown at GET /metrics (Prometheus text format).
"""
import json
import logging
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone

# ----------------------------------------------------------------------------
# 1) What is allowed in the logs
# ----------------------------------------------------------------------------
# Only these request fields may appear in a log line (all are plain car attributes).
LOGGABLE_FIELDS = {"brand", "model_year", "mileage", "fuel_type", "transmission", "hp",
                   "engine_displacement", "is_v_engine", "accident", "clean_title"}

# If a client sends keys like these, we log the KEY with "[REDACTED]" and never the value.
SENSITIVE_KEYS = {"password", "token", "authorization", "api_key", "secret", "cookie",
                  "email", "phone", "name", "address", "vin", "ssn"}


def safe_payload(payload):
    """Return a copy of the request body that is safe to write to a log."""
    if not isinstance(payload, dict):
        return None
    safe = {}
    for key, value in payload.items():
        if str(key).lower() in SENSITIVE_KEYS:
            safe[key] = "[REDACTED]"
        elif key in LOGGABLE_FIELDS:
            safe[key] = value[:50] if isinstance(value, str) else value     # never log huge strings
        # any other key is silently dropped (allow-list)
    return safe


# ----------------------------------------------------------------------------
# 2) JSON logging to stdout
# ----------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        entry = {
            "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def setup_logging(level="INFO"):
    logger = logging.getLogger("inference")
    logger.setLevel(level.upper())
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)       # containers: log to stdout, Docker collects it
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ----------------------------------------------------------------------------
# 3) In-app metrics (Prometheus text format, no extra library)
# ----------------------------------------------------------------------------
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)    # seconds


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()                  # the server handles several requests at once
        self._started = time.time()
        self.requests = defaultdict(int)               # (method, route, status) -> count
        self.latency_buckets = defaultdict(lambda: [0] * len(LATENCY_BUCKETS))
        self.latency_sum = defaultdict(float)          # route -> total seconds
        self.latency_count = defaultdict(int)          # route -> number of requests
        self.predictions = defaultdict(int)            # model name -> predictions served

    def observe_request(self, method, route, status, seconds):
        with self._lock:
            self.requests[(method, route, status)] += 1
            self.latency_sum[route] += seconds
            self.latency_count[route] += 1
            for i, upper in enumerate(LATENCY_BUCKETS):
                if seconds <= upper:
                    self.latency_buckets[route][i] += 1
                    break

    def count_prediction(self, model_name):
        with self._lock:
            self.predictions[model_name] += 1

    def render(self):
        """Prometheus text exposition format."""
        lines = []
        with self._lock:
            lines += ["# HELP http_requests_total Total HTTP requests.", "# TYPE http_requests_total counter"]
            for (method, route, status), n in sorted(self.requests.items()):
                lines.append(f'http_requests_total{{method="{method}",route="{route}",status="{status}"}} {n}')

            lines += ["# HELP http_request_duration_seconds Request latency.", "# TYPE http_request_duration_seconds histogram"]
            for route in sorted(self.latency_count):
                cumulative = 0
                for upper, n in zip(LATENCY_BUCKETS, self.latency_buckets[route]):
                    cumulative += n
                    lines.append(f'http_request_duration_seconds_bucket{{route="{route}",le="{upper}"}} {cumulative}')
                lines.append(f'http_request_duration_seconds_bucket{{route="{route}",le="+Inf"}} {self.latency_count[route]}')
                lines.append(f'http_request_duration_seconds_sum{{route="{route}"}} {self.latency_sum[route]:.6f}')
                lines.append(f'http_request_duration_seconds_count{{route="{route}"}} {self.latency_count[route]}')

            lines += ["# HELP predictions_total Predictions served.", "# TYPE predictions_total counter"]
            for model, n in sorted(self.predictions.items()):
                lines.append(f'predictions_total{{model="{model}"}} {n}')

            lines += ["# HELP app_uptime_seconds Seconds since the service started.", "# TYPE app_uptime_seconds gauge",
                      f"app_uptime_seconds {time.time() - self._started:.0f}"]
        return "\n".join(lines) + "\n"
