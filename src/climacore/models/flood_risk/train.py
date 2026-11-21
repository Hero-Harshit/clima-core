"""
Supervised District Flood Disaster & Inundation Risk Classifier.
Trained on 44 years of IIT Delhi India Flood Inventory (IFI) disaster records (1980-2023)
aligned with IMD gridded daily precipitation, ERA5 multi-layer soil moisture,
and OpenStreetMap drainage/bridge/transit exposure across 50 Indian cities.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
)

from climacore.common.logging import get_logger
from climacore.config.loader import get_project_root

logger = get_logger("climacore.models.flood_risk.train")

FLOOD_FEATURES: List[str] = [
    "rainfall_mm",
    "rain_rolling_3d",
    "rain_rolling_7d",
    "rain_rolling_14d",
    "rain_rolling_30d",
    "rain_max_3d",
    "api_14d",
    "precipitation_hours",
    "et0_fao_evapotranspiration",
    "wind_gusts_10m_max",
    "soil_moisture_0_to_7cm_mean",
    "soil_moisture_7_to_28cm_mean",
    "topsoil_moisture_anomaly",
    "rootzone_moisture_anomaly",
    "soil_moisture_deficit_index",
    "drainage_canals_15km",
    "bridges_15km",
    "transit_stations_15km",
    "population_15km",
    "dense_cluster_pop_15km",
]
TARGET_COL = "flood_disaster_event"


def load_flood_training_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the aligned flood history dataset."""
    if data_path is None:
        root = get_project_root()
        data_path = root / "data" / "processed" / "floods" / "city_flood_history_aligned.parquet"

    if not data_path.exists():
        from climacore.data.sources.ifi_floods import build_aligned_flood_dataset
        logger.info("Aligned flood dataset not found. Generating now...")
        return build_aligned_flood_dataset(output_parquet=data_path)

    logger.info(f"Loading aligned flood dataset from {data_path}...")
    return pd.read_parquet(data_path)


def train_flood_risk_model(
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train and evaluate the LightGBM Flood Disaster Risk Model."""
    root = get_project_root()
    if output_dir is None:
        output_dir = root / "artifacts" / "models" / "flood_risk" / "v0.1.0"
    if report_path is None:
        report_path = root / "reports" / "evaluation" / "flood_risk_evaluation_report.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_flood_training_data(data_path)

    # Chronological temporal split
    train_mask = df["date"] <= "2018-12-31"
    val_mask = (df["date"] >= "2019-01-01") & (df["date"] <= "2021-12-31")
    test_mask = df["date"] >= "2022-01-01"

    X_train, y_train = df.loc[train_mask, FLOOD_FEATURES], df.loc[train_mask, TARGET_COL]
    X_val, y_val = df.loc[val_mask, FLOOD_FEATURES], df.loc[val_mask, TARGET_COL]
    X_test, y_test = df.loc[test_mask, FLOOD_FEATURES], df.loc[test_mask, TARGET_COL]

    logger.info(f"Training split: {len(X_train):,} rows ({y_train.sum():,} positive flood-days)")
    logger.info(f"Validation split: {len(X_val):,} rows ({y_val.sum():,} positive flood-days)")
    logger.info(f"Test split: {len(X_test):,} rows ({y_test.sum():,} positive flood-days)")

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 250,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "is_unbalance": True,
        "random_state": 42,
        "n_jobs": 4,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    # Predictions
    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred_default = (test_prob >= 0.5).astype(int)
    # Calibrated decision threshold for extreme imbalance
    optimal_thresh = 0.35
    test_pred_calibrated = (test_prob >= optimal_thresh).astype(int)

    # Metrics
    test_roc_auc = float(roc_auc_score(y_test, test_prob))
    test_pr_auc = float(average_precision_score(y_test, test_prob))
    test_f1 = float(f1_score(y_test, test_pred_calibrated))
    test_precision = float(precision_score(y_test, test_pred_calibrated, zero_division=0))
    test_recall = float(recall_score(y_test, test_pred_calibrated))
    test_brier = float(brier_score_loss(y_test, test_prob))

    # Naive baseline (rain >= 64.5 mm threshold)
    rain_baseline_pred = (df.loc[test_mask, "rainfall_mm"] >= 64.5).astype(int)
    base_roc = float(roc_auc_score(y_test, df.loc[test_mask, "rainfall_mm"]))
    base_f1 = float(f1_score(y_test, rain_baseline_pred))
    base_precision = float(precision_score(y_test, rain_baseline_pred, zero_division=0))
    base_recall = float(recall_score(y_test, rain_baseline_pred))

    feature_importances = dict(zip(FLOOD_FEATURES, [int(x) for x in model.feature_importances_]))
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

    # Save booster
    booster_path = output_dir / "flood_risk_v1.booster"
    model.booster_.save_model(str(booster_path))
    logger.info(f"Saved trained booster to {booster_path}")

    # Compute hash
    hasher = hashlib.sha256()
    with open(booster_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    model_sha256 = hasher.hexdigest()

    metadata = {
        "model_name": "ClimaCore District Flood Disaster Risk Classifier",
        "version": "v0.1.0",
        "architecture": "LightGBM Binary GBDT with Antecedent Hydrology & Infrastructure Features",
        "sha256": model_sha256,
        "training_period": "1980-2018",
        "validation_period": "2019-2021",
        "test_period": "2022-2023",
        "total_training_samples": len(X_train),
        "total_test_samples": len(X_test),
        "features": FLOOD_FEATURES,
        "feature_importances": dict(sorted_features),
        "metrics": {
            "test_roc_auc": round(test_roc_auc, 4),
            "test_pr_auc": round(test_pr_auc, 4),
            "test_f1": round(test_f1, 4),
            "test_precision": round(test_precision, 4),
            "test_recall": round(test_recall, 4),
            "test_brier_score": round(test_brier, 4),
            "decision_threshold": optimal_thresh,
        },
        "baseline_comparison": {
            "naive_rainfall_threshold_roc_auc": round(base_roc, 4),
            "naive_rainfall_threshold_f1": round(base_f1, 4),
            "naive_rainfall_threshold_precision": round(base_precision, 4),
            "naive_rainfall_threshold_recall": round(base_recall, 4),
            "roc_auc_lift_pct": round(((test_roc_auc - base_roc) / base_roc) * 100, 2),
        }
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model metadata saved to {metadata_path}")
    logger.info(f"Evaluation report saved to {report_path}")

    return metadata


if __name__ == "__main__":
    rep = train_flood_risk_model()
    print("Flood Risk Model Training Completed Successfully!")
    print(f"Test ROC-AUC: {rep['metrics']['test_roc_auc']} (Lift vs baseline: +{rep['baseline_comparison']['roc_auc_lift_pct']}%)")
    print(f"Test Recall:  {rep['metrics']['test_recall']} at threshold {rep['metrics']['decision_threshold']}")
