"""
features.py - turn a human-friendly request into the exact feature table the saved model expects.

WHY THIS FILE EXISTS
--------------------
The saved model (best_model.pkl) does:   brand encoding -> scaling -> estimator -> price in dollars.
But in the training notebook, some steps happened BEFORE the model: parsing the car's age, mileage per year,
mileage/age bins, label encoding of fuel type / transmission, etc.
Those steps must be repeated for every new request, in exactly the same way. This file does that,
using the definitions the notebook saved in model_metadata.json ("feature_preparation").
"""
import pandas as pd

AGE_LABELS     = ["New", "Mid", "Old", "Very Old"]            # same labels as in the notebook
MILEAGE_LABELS = ["Low", "Medium", "High", "Very High"]

FUEL_SYNONYMS = {"PLUG-IN HYBRID": "HYBRID", "NOT SUPPORTED": "OTHER", "–": "OTHER"}   # same cleaning as the notebook

REQUIRED_FIELDS = ["brand", "model_year", "mileage", "fuel_type", "transmission",
                   "hp", "engine_displacement", "is_v_engine", "accident", "clean_title"]


class ValidationError(ValueError):
    """Raised when the request body is not acceptable -> the API answers HTTP 400."""
    def __init__(self, errors):
        super().__init__("; ".join(errors))
        self.errors = errors


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "yes", "1"):
        return True
    if isinstance(value, str) and value.strip().lower() in ("false", "no", "0"):
        return False
    return None


def _to_number(value):
    if isinstance(value, bool):          # True/False are not valid numbers here
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate(payload, meta, known_brands):
    """Check the request. Returns (clean_values, warnings) or raises ValidationError."""
    errors, warnings, clean = [], [], {}
    fp = meta["feature_preparation"]
    ref_year = fp.get("vehicle_age_reference_year", 2025)

    if not isinstance(payload, dict):
        raise ValidationError(["Request body must be a JSON object, e.g. {\"brand\": \"BMW\", ...}"])

    missing = [f for f in REQUIRED_FIELDS if f not in payload or payload[f] in (None, "")]
    if missing:
        raise ValidationError([f"Missing field(s): {', '.join(missing)}"])

    # brand: free text; unknown brands are allowed (the model falls back to the average) but we warn
    brand = str(payload["brand"]).strip()
    canonical = {b.lower(): b for b in known_brands}
    if brand.lower() in canonical:
        brand = canonical[brand.lower()]
    else:
        warnings.append(f"Brand '{brand}' was not seen in training - the model uses the average brand price level.")
    clean["brand"] = brand

    # numbers with sensible ranges
    for key, lo, hi, integer in [("model_year", 1980, ref_year, True), ("mileage", 0, 1_000_000, False),
                                 ("hp", 20, 1500, False), ("engine_displacement", 0.5, 10, False)]:
        num = _to_number(payload[key])
        if num is None:
            errors.append(f"'{key}' must be a number")
        elif not lo <= num <= hi:
            errors.append(f"'{key}' must be between {lo} and {hi}")
        else:
            clean[key] = int(num) if integer else num

    # categories: must be one of the values the model was trained with
    fuel = str(payload["fuel_type"]).strip().upper()
    fuel = FUEL_SYNONYMS.get(fuel, fuel)
    if fuel not in fp["label_encoder_classes"]["fuel_type"]:
        errors.append(f"'fuel_type' must be one of {fp['label_encoder_classes']['fuel_type']}")
    clean["fuel_type"] = fuel

    trans = str(payload["transmission"]).strip().upper()
    if trans not in fp["label_encoder_classes"]["transmission"]:
        errors.append(f"'transmission' must be one of {fp['label_encoder_classes']['transmission']}")
    clean["transmission"] = trans

    # true/false fields
    for key in ("is_v_engine", "accident", "clean_title"):
        b = _to_bool(payload[key])
        if b is None:
            errors.append(f"'{key}' must be true or false")
        clean[key] = b

    if errors:
        raise ValidationError(errors)
    return clean, warnings


def _bin_index(value, edges):
    """Same rule as pd.qcut: bins are (e0,e1], (e1,e2], ... Values outside the training range go to the first/last bin."""
    return sum(1 for e in edges[1:-1] if e < value)


def _dummies(prefix, labels, edges, value):
    """One-hot columns with drop_first=True, exactly like pd.get_dummies in the notebook."""
    idx = _bin_index(value, edges)
    return {f"{prefix}_{label}": int(i == idx) for i, label in enumerate(labels) if i > 0}


def build_features(clean, meta):
    """Clean request -> one-row DataFrame with the model's exact input columns (names, order, dtypes)."""
    fp = meta["feature_preparation"]
    classes = fp["label_encoder_classes"]
    ref_year = fp.get("vehicle_age_reference_year", 2025)

    age = ref_year - clean["model_year"]
    mileage_per_year = clean["mileage"] / age if age > 0 else clean["mileage"]

    row = {
        "brand": clean["brand"],                                              # encoded INSIDE the model pipeline
        "fuel_type": classes["fuel_type"].index(clean["fuel_type"]),          # LabelEncoder: position in the class list
        "transmission": classes["transmission"].index(clean["transmission"]),
        "clean_title": int(clean["clean_title"]),
        "hp": float(clean["hp"]),
        "engine displacement": float(clean["engine_displacement"]),
        "is_v_engine": classes["is_v_engine"].index(str(bool(clean["is_v_engine"]))),
        "Accident_Impact": int(clean["accident"]),
        "Vehicle_Age": age,
        "Mileage_per_Year": mileage_per_year,
    }
    row.update(_dummies("Age", AGE_LABELS, fp["vehicle_age_bin_edges"], age))
    row.update(_dummies("Milage", MILEAGE_LABELS, fp["mileage_bin_edges"], clean["mileage"]))

    columns = meta["model_input"]["feature_columns"]
    frame = pd.DataFrame([row])[columns]                                      # exact column order of training

    for col, dtype in meta["model_input"]["dtypes"].items():                  # same numeric dtypes as training
        if dtype.startswith(("int", "float")):
            frame[col] = frame[col].astype(dtype)
    return frame
