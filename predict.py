"""Serve the laboratory quality prediction model over HTTP with Flask.

The model artifact is produced by ``train.py``. This script loads it and
exposes a small JSON API used by plant operators and by the deployment
manifests.

Run locally:
    python predict.py                      # dev server on :8000
    gunicorn --bind 0.0.0.0:8000 --workers 2 --threads 4 predict:app
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest

from src.data import FEATURE_NAMES, build_features

app = Flask(__name__)

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "artifacts/model.joblib"))
MAX_BATCH_SIZE = 500

_BUNDLE: dict | None = None


def get_bundle() -> dict:
    """Load the model artifact once and cache it."""
    global _BUNDLE
    if _BUNDLE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {MODEL_PATH}. Run: python train.py"
            )
        _BUNDLE = joblib.load(MODEL_PATH)
    return _BUNDLE


@app.before_request
def _require_model() -> None:
    """Fail fast if the artifact is missing rather than serving 500s later."""
    get_bundle()


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "type": str,
    "air_temperature": float,
    "process_temperature": float,
    "rotational_speed": float,
    "torque": float,
    "tool_wear": float,
}

#: Maps the API field names onto the dataset's original column names.
API_TO_DATASET = {
    "type": "Type",
    "air_temperature": "Air temperature",
    "process_temperature": "Process temperature",
    "rotational_speed": "Rotational speed",
    "torque": "Torque",
    "tool_wear": "Tool wear",
}


def payload_to_frame(payload: dict) -> pd.DataFrame:
    """Validate one request payload and turn it into a one-row feature frame.

    Uses the same :func:`build_features` as training, so the served model
    always receives the columns it was fitted on.
    """
    if not isinstance(payload, dict):
        raise BadRequest("Request body must be a JSON object")

    missing = [key for key in REQUIRED_FIELDS if key not in payload]
    if missing:
        raise BadRequest(f"Missing required fields: {', '.join(missing)}")

    row: dict = {}
    for key, caster in REQUIRED_FIELDS.items():
        value = payload[key]
        if caster is str:
            row[API_TO_DATASET[key]] = str(value).strip().upper()
        else:
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise BadRequest(f"Field '{key}' must be a number, got {value!r}")
            if not math.isfinite(number):
                raise BadRequest(f"Field '{key}' must be a finite number, got {value!r}")
            row[API_TO_DATASET[key]] = number

    frame = build_features(pd.DataFrame([row]))
    if list(frame.columns) != list(FEATURE_NAMES):
        raise BadRequest("Internal feature construction mismatch")
    return frame


def predict_one(payload: dict) -> dict:
    """Score a single product and build the response body."""
    bundle = get_bundle()
    frame = payload_to_frame(payload)

    expected = bundle["feature_names"]
    if list(frame.columns) != list(expected):
        raise BadRequest(
            "Feature columns do not match the trained model "
            f"({list(frame.columns)} != {list(expected)})"
        )

    prob_fail = float(bundle["pipeline"].predict_proba(frame)[0, 1])
    threshold = float(bundle["threshold"])
    quality_label = int(prob_fail >= threshold)  # 1 = Fail, 0 = Pass

    row = frame.iloc[0]
    return {
        "quality_status": "Fail" if quality_label == 1 else "Pass",
        "quality_label": quality_label,
        "machine_failure_probability": round(prob_fail, 6),
        "pass_probability": round(1.0 - prob_fail, 6),
        "threshold": threshold,
        "explanation": {
            "air_temp_oos": int(row["air_temp_oos"]),
            "proc_temp_oos": int(row["proc_temp_oos"]),
            "torque_oos": int(row["torque_oos"]),
            "rot_speed_oos": int(row["rot_speed_oos"]),
        },
    }


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@app.get("/")
def index():
    bundle = get_bundle()
    return jsonify(
        {
            "service": "Laboratory Quality Prediction System",
            "description": (
                "Predicts whether a manufactured product passes or fails "
                "quality inspection."
            ),
            "model": bundle.get("model_name"),
            "target_definition": bundle.get("target_definition"),
            "endpoints": {
                "GET /health": "liveness / readiness probe",
                "POST /predict": "predict one product",
                "POST /predict/batch": "predict a list of products (max 500)",
            },
            "example_request": {
                "type": "L",
                "air_temperature": 300.5,
                "process_temperature": 310.2,
                "rotational_speed": 1500,
                "torque": 40.1,
                "tool_wear": 12,
            },
        }
    )


@app.get("/health")
def health():
    bundle = get_bundle()
    return jsonify(
        {
            "status": "ok",
            "model": bundle.get("model_name"),
            "sklearn_version": bundle.get("sklearn_version"),
            "trained_at": bundle.get("trained_at"),
            "threshold": bundle.get("threshold"),
            "target_definition": bundle.get("target_definition"),
            "features": bundle.get("feature_names"),
        }
    )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        raise BadRequest("Request body must be valid JSON")
    return jsonify(predict_one(payload))


@app.post("/predict/batch")
def predict_batch():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or "products" not in payload:
        raise BadRequest("Request body must be a JSON object with a 'products' list")

    products = payload["products"]
    if not isinstance(products, list):
        raise BadRequest("'products' must be a list")
    if not products:
        raise BadRequest("'products' must not be empty")
    if len(products) > MAX_BATCH_SIZE:
        raise BadRequest(f"'products' may contain at most {MAX_BATCH_SIZE} items")

    return jsonify(
        {"count": len(products), "predictions": [predict_one(p) for p in products]}
    )


@app.errorhandler(BadRequest)
def handle_bad_request(error):
    return jsonify({"error": "bad_request", "message": error.description}), 400


@app.errorhandler(FileNotFoundError)
def handle_missing_model(error):
    return jsonify({"error": "model_unavailable", "message": str(error)}), 503


@app.errorhandler(404)
def handle_not_found(error):
    return jsonify({"error": "not_found", "message": "Unknown endpoint"}), 404


if __name__ == "__main__":
    get_bundle()  # fail at startup, not on the first request
    port = int(os.environ.get("PORT", 8000))
    print(f"Serving on http://0.0.0.0:{port}  (model: {MODEL_PATH})")
    app.run(host="0.0.0.0", port=port)
