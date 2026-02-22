"""
FastAPI backend for the DataQuest Insurance Bundle Predictor.

Endpoints:
    POST /predict    Upload a CSV and get predictions back
    GET  /info       Model metadata (size, features, classes)
    GET  /metrics    Latest training metrics (F1, latency, model size)
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from src.solution import load_model, predict, preprocess

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DataQuest  Insurance Bundle Predictor",
    version="1.0.0",
    description="Predict the optimal insurance coverage bundle for each user.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model.joblib"


# ── Lazy-loaded model singleton ──────────────────────────────────────────────
_model = None


def _get_model():
    """Load the model once and cache it in memory."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="model.joblib not found. Train the model first.")
        _model = load_model()
    return _model


# ── Helpers ──────────────────────────────────────────────────────────────────
def _read_csv_from_upload(file: UploadFile) -> pd.DataFrame:
    """Read an uploaded file into a DataFrame, with validation."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")
    try:
        contents = file.file.read()
        return pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# POST /predict
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/predict", summary="Run predictions on an uploaded CSV")
async def run_predict(file: UploadFile = File(...)):
    """
    Accepts a CSV with the same schema as train.csv (minus the target column).
    Returns a CSV with columns: `User_ID`, `Purchased_Coverage_Bundle`.
    """
    df = _read_csv_from_upload(file)

    # Validate required column
    if "User_ID" not in df.columns:
        raise HTTPException(status_code=422, detail="CSV must contain a 'User_ID' column.")

    model = _get_model()

    # Preprocess → predict → time it
    start = time.perf_counter()
    processed = preprocess(df)
    results = predict(processed, model)
    duration = time.perf_counter() - start

    # Return CSV as downloadable response
    output_buf = io.StringIO()
    results.to_csv(output_buf, index=False)
    output_buf.seek(0)

    return StreamingResponse(
        io.BytesIO(output_buf.getvalue().encode()),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=predictions.csv",
            "X-Prediction-Time-Seconds": f"{duration:.4f}",
            "X-Rows-Processed": str(len(df)),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /info
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/info", summary="Model and system information")
async def model_info():
    """
    Returns metadata about the loaded model:
    size on disk, number of trees, feature names, and target classes.
    """
    model = _get_model()

    size_bytes = MODEL_PATH.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    # LightGBM-specific attributes (graceful fallback for other models)
    info = {
        "model_file": MODEL_PATH.name,
        "model_size_mb": round(size_mb, 3),
        "model_type": type(model).__name__,
    }

    if hasattr(model, "n_estimators"):
        info["n_estimators"] = int(model.n_estimators)
    if hasattr(model, "n_features_in_"):
        info["n_features"] = int(model.n_features_in_)
    if hasattr(model, "feature_name_"):
        info["feature_names"] = model.feature_name_
    if hasattr(model, "classes_"):
        info["classes"] = [int(c) for c in model.classes_]
    if hasattr(model, "best_iteration_"):
        info["best_iteration"] = int(model.best_iteration_)

    return info


# ─────────────────────────────────────────────────────────────────────────────
# GET /metrics
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/metrics", summary="Latest model performance metrics")
async def model_metrics():
    """
    Loads the training data, runs predictions, and computes live metrics:
    macro-F1, model size penalty, latency penalty, and the final competition score.

    Score formula: final = macro_f1 × max(0.5, 1 − size_mb/200) × max(0.5, 1 − duration_s/10)
    """
    model = _get_model()
    train_path = BASE_DIR / "data" / "train.csv"

    if not train_path.exists():
        raise HTTPException(status_code=404, detail="data/train.csv not found.")

    raw = pd.read_csv(train_path)

    if "Purchased_Coverage_Bundle" not in raw.columns:
        raise HTTPException(status_code=422, detail="train.csv missing target column.")

    y_true = raw["Purchased_Coverage_Bundle"].values

    # Timed prediction
    start = time.perf_counter()
    processed = preprocess(raw)
    preds_df = predict(processed, model)
    duration = time.perf_counter() - start

    y_pred = preds_df["Purchased_Coverage_Bundle"].values

    # --- Competition scoring ---
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
    )

    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    accuracy = float(accuracy_score(y_true, y_pred))

    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    size_penalty = max(0.5, 1 - size_mb / 200)
    time_penalty = max(0.5, 1 - duration / 10)
    final_score = macro_f1 * size_penalty * time_penalty

    # Per-class report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(accuracy, 4),
        "model_size_mb": round(size_mb, 3),
        "prediction_time_s": round(duration, 3),
        "size_penalty": round(size_penalty, 4),
        "time_penalty": round(time_penalty, 4),
        "final_score": round(final_score, 4),
        "total_samples": len(y_true),
        "per_class_report": report,
    }
