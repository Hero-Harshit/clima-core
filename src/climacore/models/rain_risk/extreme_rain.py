"""
Extreme Heavy Rainfall Classifier for Clima-Core.
Trains a cost-sensitive (class-weighted) LightGBM classifier specifically tuned
for detecting rare, high-impact IMD Moderate/Heavy rain events (>= 15.6 mm).
Evaluates Precision, Recall, and PR-AUC on the 2026 holdout test set.
"""

from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime, timezone
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from climacore.common.logging import get_logger
from climacore.common.provenance import compute_sha256, get_file_metadata
from climacore.config.loader import get_project_root
from climacore.evaluation.splits import chronological_train_val_test_split

logger = get_logger("climacore.models.rain_risk.extreme_rain")


def train_extreme_rain_classifier(
    threshold_mm: float = 15.6,
    val_start: str = "2025-01-01",
    test_start: str = "2026-01-01",
) -> Dict[str, Any]:
    """Train class-weighted LightGBM classifier for rain >= threshold_mm."""
    root = get_project_root()
    art_path = root / "artifacts" / "models" / "rain_risk" / "extreme_rain"
    rep_path = root / "reports" / "evaluation"
    art_path.mkdir(parents=True, exist_ok=True)
    rep_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading RainRisk data for extreme rain classification (threshold >= {threshold_mm} mm)...")
    df_rain = pd.read_parquet(root / "data" / "processed" / "rain_risk_train_ready.parquet")

    # Define binary target for extreme threshold
    target_col = f"target_extreme_rain_{int(threshold_mm)}"
    df_rain[target_col] = (df_rain["target_precipitation_next_day"] >= threshold_mm).astype(int)

    with open(root / "data" / "processed" / "rain_risk_feature_schema.json") as f:
        rain_feats = json.load(f)["feature_columns"]

    train_df, val_df, test_df = chronological_train_val_test_split(df_rain, "date", val_start, test_start)

    y_train = train_df[target_col].to_numpy()
    y_val = val_df[target_col].to_numpy()
    y_test = test_df[target_col].to_numpy()

    # Calculate class weight ratio (neg / pos) to counteract extreme imbalance
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    scale_weight = float(neg_count / max(1, pos_count))
    logger.info(f"Class imbalance: {pos_count:,} positive vs {neg_count:,} negative (scale_pos_weight: {scale_weight:.2f})")

    lgb_params = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "n_estimators": 800,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": min(10.0, scale_weight),  # Moderated class weight to balance precision and recall
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 4,
    }

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        train_df[rain_feats], y_train,
        eval_set=[(val_df[rain_feats], y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    best_iter = int(model.best_iteration_)
    logger.info(f"Extreme Rain Classifier converged at iteration {best_iter}.")

    # Evaluate on test set
    test_probs = model.predict_proba(test_df[rain_feats])[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    prec = round(float(precision_score(y_test, test_preds, zero_division=0)), 4)
    rec = round(float(recall_score(y_test, test_preds, zero_division=0)), 4)
    f1 = round(float(f1_score(y_test, test_preds, zero_division=0)), 4)
    pr_auc = round(float(average_precision_score(y_test, test_probs)), 4)
    roc_auc = round(float(roc_auc_score(y_test, test_probs)), 4)
    cm = confusion_matrix(y_test, test_preds).tolist()

    logger.info(f"Extreme Rain Test Metrics: Recall={rec*100:.1f}%, Precision={prec*100:.1f}%, PR-AUC={pr_auc}, ROC-AUC={roc_auc}")

    # Save artifact
    model_file = art_path / "extreme_rain_classifier.txt"
    model.booster_.save_model(str(model_file))

    results = {
        "model_name": "climacore_extreme_rain_classifier",
        "threshold_mm": threshold_mm,
        "classification_target": f"precipitation >= {threshold_mm} mm",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_iteration": best_iter,
        "test_metrics": {
            "recall": rec,
            "precision": prec,
            "f1_score": f1,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "actual_positive_events_in_2026": int(np.sum(y_test == 1)),
            "detected_events": int(np.sum(test_preds == 1)),
        },
        "artifact_file": str(model_file.name),
    }

    out_json = rep_path / "extreme_rain_evaluation_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    train_extreme_rain_classifier()
