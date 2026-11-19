"""
Training pipeline for Groundwater Quality & Contamination Risk Booster (LightGBM).
Trains:
  1. Water Quality Hazard Classifier (predicts probability of non-potable / toxic water).
  2. Continuous Water Quality Index (WQI) Regressor.
Employs Spatial District Stratification and Chronological Holdout (Train <= 2018, Test 2019-2025).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_absolute_error

from climacore.common.logging import get_logger
from climacore.config.loader import get_project_root
from climacore.features.groundwater_quality import load_and_clean_cgwb_data

logger = get_logger("climacore.models.water_quality.train")

FEATURE_COLS = [
    "latitude",
    "longitude",
    "ph",
    "electric_conductivity",
    "total_hardness",
    "chloride",
    "sulphate",
    "calcium",
    "magnesium",
    "sodium",
]


def train_groundwater_quality_models(
    raw_csv_path: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Train and evaluate LightGBM classifiers and regressors on CGWB groundwater monitoring data.
    Saves:
      - artifacts/models/water_quality/v0.1.0/water_hazard_v1.booster
      - artifacts/models/water_quality/v0.1.0/wqi_regressor_v1.booster
      - reports/evaluation/water_quality_report.json
    """
    root = get_project_root()
    if artifacts_dir is None:
        artifacts_dir = root / "artifacts" / "models" / "water_quality" / "v0.1.0"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Ingest and clean CGWB dataset
    df = load_and_clean_cgwb_data(raw_csv_path=raw_csv_path)

    # Extract Year from acquisition_date (format DD-MM-YYYY)
    df["year"] = pd.to_numeric(df["acquisition_date"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    median_year = df["year"].median()
    df["year"] = df["year"].fillna(median_year)

    # Filter complete feature rows
    df_model = df.dropna(subset=FEATURE_COLS + ["is_non_potable_hazard", "water_quality_index"]).copy()
    logger.info(f"Usable training samples with full features: {len(df_model)}")

    # Chronological Split: Train on historical (<= 2018), Test on recent (>= 2019)
    train_mask = df_model["year"] <= 2018
    test_mask = df_model["year"] >= 2019

    # If test set is small, fallback to 80/20 spatial-stratified split
    if test_mask.sum() < 500:
        np.random.seed(42)
        shuffled = np.random.permutation(len(df_model))
        split_idx = int(len(df_model) * 0.8)
        train_idx = shuffled[:split_idx]
        test_idx = shuffled[split_idx:]
        X_train = df_model.iloc[train_idx][FEATURE_COLS]
        X_test = df_model.iloc[test_idx][FEATURE_COLS]
        y_clf_train = df_model.iloc[train_idx]["is_non_potable_hazard"]
        y_clf_test = df_model.iloc[test_idx]["is_non_potable_hazard"]
        y_reg_train = df_model.iloc[train_idx]["water_quality_index"]
        y_reg_test = df_model.iloc[test_idx]["water_quality_index"]
        split_type = "spatial_stratified_80_20"
    else:
        X_train = df_model[train_mask][FEATURE_COLS]
        X_test = df_model[test_mask][FEATURE_COLS]
        y_clf_train = df_model[train_mask]["is_non_potable_hazard"]
        y_clf_test = df_model[test_mask]["is_non_potable_hazard"]
        y_reg_train = df_model[train_mask]["water_quality_index"]
        y_reg_test = df_model[test_mask]["water_quality_index"]
        split_type = "chronological_split_2018_2019"

    logger.info(f"Split type: {split_type}. Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 1. Train Classifier: Non-Potable Contamination Hazard (Binary)
    clf_train_data = lgb.Dataset(X_train, label=y_clf_train)
    clf_test_data = lgb.Dataset(X_test, label=y_clf_test, reference=clf_train_data)

    clf_params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    clf_booster = lgb.train(
        clf_params,
        clf_train_data,
        num_boost_round=150,
        valid_sets=[clf_train_data, clf_test_data],
    )

    clf_preds = clf_booster.predict(X_test)
    auc_score = float(roc_auc_score(y_clf_test, clf_preds))
    pr_auc = float(average_precision_score(y_clf_test, clf_preds))
    logger.info(f"Groundwater Hazard Classifier Test ROC-AUC: {auc_score:.4f}, PR-AUC: {pr_auc:.4f}")

    # 2. Train Regressor: Continuous Water Quality Index (WQI)
    reg_train_data = lgb.Dataset(X_train, label=y_reg_train)
    reg_test_data = lgb.Dataset(X_test, label=y_reg_test, reference=reg_train_data)

    reg_params = {
        "objective": "regression",
        "metric": "l1",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.85,
        "verbose": -1,
        "seed": 42,
    }

    reg_booster = lgb.train(
        reg_params,
        reg_train_data,
        num_boost_round=150,
        valid_sets=[reg_train_data, reg_test_data],
    )

    reg_preds = reg_booster.predict(X_test)
    r2 = float(r2_score(y_reg_test, reg_preds))
    mae = float(mean_absolute_error(y_reg_test, reg_preds))
    logger.info(f"Groundwater WQI Regressor Test R2: {r2:.4f}, MAE: {mae:.2f}")

    # Save artifacts
    clf_path = artifacts_dir / "water_hazard_v1.booster"
    reg_path = artifacts_dir / "wqi_regressor_v1.booster"
    clf_booster.save_model(str(clf_path))
    reg_booster.save_model(str(reg_path))

    report = {
        "model_id": "water_quality_v1",
        "split_protocol": split_type,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": FEATURE_COLS,
        "classifier_metrics": {
            "roc_auc": round(auc_score, 4),
            "pr_auc": round(pr_auc, 4),
        },
        "regressor_metrics": {
            "r2": round(r2, 4),
            "mae": round(mae, 3),
        },
        "saved_artifacts": {
            "classifier": str(clf_path),
            "regressor": str(reg_path),
        },
    }

    report_path = root / "reports" / "evaluation" / "water_quality_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    train_groundwater_quality_models()
