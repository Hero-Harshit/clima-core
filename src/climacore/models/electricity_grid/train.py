"""
Heat-to-Electricity Grid Stress & Blackout Hazard Intelligence Model.
Trained on 8 years (2013-2020) of state-level electricity grid telemetry (POSOCO / CEA)
aligned with UTCI thermal climate index, cooling degree days, and 2m temperature
across Indian urban centers.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
)

from climacore.common.logging import get_logger
from climacore.config.loader import get_project_root

logger = get_logger("climacore.models.electricity_grid.train")

GRID_FEATURES: List[str] = [
    "city",
    "temp_min_c",
    "temp_mean_c",
    "temp_max_c",
    "dewpoint_mean_c",
    "utci_mean_c",
    "utci_max_c",
    "cooling_degree_days",
    "heating_degree_days",
    "is_tropical_night",
    "diurnal_temp_range",
    "solar_radiation_mj",
    "cloud_cover_pct",
    "day_of_week",
    "is_weekend",
    "month",
    "population_15km",
    "dense_cluster_pop_15km",
]


def load_electricity_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """Load aligned electricity grid dataset."""
    if data_path is None:
        root = get_project_root()
        data_path = root / "data" / "processed" / "electricity" / "city_daily_electricity_stress.parquet"

    if not data_path.exists():
        from climacore.data.sources.electricity_demand import build_aligned_electricity_dataset
        return build_aligned_electricity_dataset(output_parquet=data_path)

    logger.info(f"Loading electricity dataset from {data_path}...")
    df = pd.read_parquet(data_path)
    return df.dropna(subset=["energy_met_mu", "peak_met_mw", "grid_stress_ratio"]).copy()


def train_electricity_grid_models(
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train peak load regressor and grid deficit hazard classifier."""
    root = get_project_root()
    if output_dir is None:
        output_dir = root / "artifacts" / "models" / "electricity_grid" / "v0.1.0"
    if report_path is None:
        report_path = root / "reports" / "evaluation" / "electricity_grid_evaluation_report.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_electricity_data(data_path)
    df["city"] = df["city"].astype("category")

    # Temporal split: 2013-2018 Train, 2019 Val, 2020 Holdout Test
    train_mask = df["date"] <= "2018-12-31"
    val_mask = (df["date"] >= "2019-01-01") & (df["date"] <= "2019-12-31")
    test_mask = df["date"] >= "2020-01-01"

    X_train, X_val, X_test = df.loc[train_mask, GRID_FEATURES], df.loc[val_mask, GRID_FEATURES], df.loc[test_mask, GRID_FEATURES]

    # Target 1: Peak Grid Load (MW)
    y_reg_train = df.loc[train_mask, "peak_met_mw"]
    y_reg_val = df.loc[val_mask, "peak_met_mw"]
    y_reg_test = df.loc[test_mask, "peak_met_mw"]

    # Target 2: Grid Deficit / Load Shedding Hazard (Binary)
    y_clf_train = df.loc[train_mask, "is_grid_deficit_day"]
    y_clf_val = df.loc[val_mask, "is_grid_deficit_day"]
    y_clf_test = df.loc[test_mask, "is_grid_deficit_day"]

    logger.info(f"Training split: {len(X_train):,} rows")
    logger.info(f"Validation split: {len(X_val):,} rows")
    logger.info(f"Test split: {len(X_test):,} rows")

    # 1. Train Peak Grid Load Regressor
    logger.info("Training Peak Grid Load Regressor (MW)...")
    reg_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 250,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": 4,
        "verbose": -1,
    }
    reg = lgb.LGBMRegressor(**reg_params)
    reg.fit(
        X_train, y_reg_train,
        eval_set=[(X_val, y_reg_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
    )

    preds_reg = reg.predict(X_test)
    test_r2 = float(r2_score(y_reg_test, preds_reg))
    test_mae = float(mean_absolute_error(y_reg_test, preds_reg))
    test_rmse = float(root_mean_squared_error(y_reg_test, preds_reg))

    # 2. Train Grid Deficit / Blackout Hazard Classifier
    logger.info("Training Grid Deficit / Blackout Hazard Classifier...")
    clf_params = {
        "objective": "binary",
        "n_estimators": 150,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": 4,
        "verbose": -1,
    }
    clf = lgb.LGBMClassifier(**clf_params)
    clf.fit(X_train, y_clf_train)

    probs_clf = clf.predict_proba(X_test)[:, 1]
    test_auc = float(roc_auc_score(y_clf_test, probs_clf))
    test_pr_auc = float(average_precision_score(y_clf_test, probs_clf))
    test_f1 = float(f1_score(y_clf_test, (probs_clf >= 0.40).astype(int), zero_division=0))
    test_recall = float(recall_score(y_clf_test, (probs_clf >= 0.40).astype(int), zero_division=0))

    # Save boosters
    reg_path = output_dir / "peak_grid_load_v1.booster"
    clf_path = output_dir / "grid_deficit_hazard_v1.booster"
    reg.booster_.save_model(str(reg_path))
    clf.booster_.save_model(str(clf_path))
    logger.info(f"Saved models to {reg_path} and {clf_path}")

    # Compute hashes
    h_reg = hashlib.sha256()
    with open(reg_path, "rb") as f:
        while chunk := f.read(65536):
            h_reg.update(chunk)

    h_clf = hashlib.sha256()
    with open(clf_path, "rb") as f:
        while chunk := f.read(65536):
            h_clf.update(chunk)

    # Feature importance
    feat_names = [f for f in GRID_FEATURES if f != "city"]
    imp_reg = dict(zip(feat_names, [int(x) for x in reg.feature_importances_[1:]]))
    imp_clf = dict(zip(feat_names, [int(x) for x in clf.feature_importances_[1:]]))

    metadata = {
        "model_suite": "ClimaCore Heat-to-Electricity Grid Stress Intelligence",
        "version": "v0.1.0",
        "regressor_sha256": h_reg.hexdigest(),
        "classifier_sha256": h_clf.hexdigest(),
        "training_period": "2013-2018",
        "validation_period": "2019",
        "test_period": "2020",
        "total_training_samples": len(X_train),
        "total_test_samples": len(X_test),
        "features": GRID_FEATURES,
        "regressor_metrics": {
            "target": "peak_met_mw",
            "test_r2": round(test_r2, 4),
            "test_mae_mw": round(test_mae, 2),
            "test_rmse_mw": round(test_rmse, 2),
            "feature_importance": dict(sorted(imp_reg.items(), key=lambda x: x[1], reverse=True)),
        },
        "classifier_metrics": {
            "target": "is_grid_deficit_day",
            "test_roc_auc": round(test_auc, 4),
            "test_pr_auc": round(test_pr_auc, 4),
            "test_f1": round(test_f1, 4),
            "test_recall": round(test_recall, 4),
            "decision_threshold": 0.40,
            "feature_importance": dict(sorted(imp_clf.items(), key=lambda x: x[1], reverse=True)),
        }
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata and report written to {report_path}")
    return metadata


if __name__ == "__main__":
    rep = train_electricity_grid_models()
    print("Electricity Grid Models Training Completed Successfully!")
    print(f"Peak Grid Load Test R2: {rep['regressor_metrics']['test_r2']} (MAE: {rep['regressor_metrics']['test_mae_mw']} MW)")
    print(f"Grid Deficit Hazard AUC: {rep['classifier_metrics']['test_roc_auc']} (Recall: {rep['classifier_metrics']['test_recall']})")
