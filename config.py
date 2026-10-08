"""
Easy configuration for the inference service.

Change a value here OR override it without touching code, using an environment variable:

    macOS / Linux :  PORT=8000 DEBUG=false python app.py
    Windows (cmd) :  set PORT=8000 && python app.py
    PowerShell    :  $env:PORT="8000"; python app.py
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---- where the trained model lives (copied from the training project: artifacts/models/) ----
MODEL_PATH    = Path(os.getenv("MODEL_PATH",    BASE_DIR / "model" / "best_model.pkl"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", BASE_DIR / "model" / "model_metadata.json"))

# ---- web server ----
HOST  = os.getenv("HOST", "127.0.0.1")          # 127.0.0.1 = only this computer; 0.0.0.0 = reachable from the network
PORT  = int(os.getenv("PORT", "5001"))           # 5001, because macOS uses 5000 for AirPlay (gives a 403)
DEBUG = os.getenv("DEBUG", "true").lower() == "true"   # True = auto-reload + detailed errors (development only)

# ---- logging ----
LOG_LEVEL    = os.getenv("LOG_LEVEL", "INFO")                         # DEBUG, INFO, WARNING, ERROR
LOG_PAYLOADS = os.getenv("LOG_PAYLOADS", "true").lower() == "true"    # log the (filtered) request fields? false = never log inputs

# ---- app text ----
APP_TITLE = "Used Car Price Predictor"
