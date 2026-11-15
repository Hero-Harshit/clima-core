"""
Evaluation metrics for Clima-Core.
Computes comprehensive regression metrics (MAE, RMSE, Bias, R2) and
AQI category classification metrics (Accuracy, Macro-F1, Severe Recall, Confusion Matrix).
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)

# CPCB official 24-hour PM2.5 breakpoints
CPCB_PM25_BANDS = [
    (0, 30, "Good"),
    (30, 60, "Satisfactory"),
    (60, 90, "Moderate"),
    (90, 120, "Poor"),
    (120, 250, "Very Poor"),
    (250, 10000, "Severe"),
]
BAND_LABELS = [b[2] for b in CPCB_PM25_BANDS]


def assign_pm25_category(value: float) -> str:
    """Classify PM2.5 concentration into official CPCB 24h category."""
    if pd.isna(value):
        return "Unknown"
    for low, high, label in CPCB_PM25_BANDS:
        if low <= value <= high:
            return label
    return "Severe" if value > 250 else "Good"


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics: MAE, RMSE, Mean Bias, R2, Median AE."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    r2 = float(r2_score(y_true, y_pred))
    mdae = float(np.median(np.abs(y_true - y_pred)))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mean_bias": round(bias, 4),
        "r2_score": round(r2, 4),
        "median_absolute_error": round(mdae, 4),
    }


def compute_aqi_category_metrics(
    y_true_values: np.ndarray,
    y_pred_values: np.ndarray,
) -> Dict[str, Any]:
    """Derive CPCB categories and compute classification evaluation metrics."""
    true_cats = [assign_pm25_category(v) for v in y_true_values]
    pred_cats = [assign_pm25_category(v) for v in y_pred_values]

    acc = float(accuracy_score(true_cats, pred_cats))
    macro_f1 = float(f1_score(true_cats, pred_cats, labels=BAND_LABELS, average="macro", zero_division=0))

    # Severe category recall (Very Poor + Severe)
    true_severe = np.array([c in ("Very Poor", "Severe") for c in true_cats])
    pred_severe = np.array([c in ("Very Poor", "Severe") for c in pred_cats])

    severe_recall = float(
        np.sum(true_severe & pred_severe) / max(1, np.sum(true_severe))
    )

    cm = confusion_matrix(true_cats, pred_cats, labels=BAND_LABELS).tolist()

    return {
        "category_accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "severe_category_recall": round(severe_recall, 4),
        "confusion_matrix_labels": BAND_LABELS,
        "confusion_matrix": cm,
    }
