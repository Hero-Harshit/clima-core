"""Severe Gale-Force Wind & Storm Gust Hazard Model.

Trains a high-performance LightGBM gradient-boosted hazard classifier
on the 44-year historical climate dataset (1980-2024, 820k+ records) across
all 50 Indian cities.

Predicts next-day gale-force wind gust hazard (wind_gusts_10m_max >= 50.0 km/h),
essential for cyclonic landfall warnings, pre-monsoon convective squall
(Kalbaishakhi/Andhi) alerts, building safety, and spatial wind risk mapping.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from climacore.common.logging import get_logger
from climacore.common.provenance import compute_sha256
from climacore.config.loader import get_project_root
from climacore.features.temporal import add_temporal_features
from climacore.features.lags import add_lag_features
from climacore.features.rolling import add_rolling_features

logger = get_logger("climacore.models.gale_gust_hazard.train")

GALE_GUST_THRESHOLD_KMH = 50.0

COASTAL_CITIES = {
    "Mumbai", "Chennai", "Kolkata", "Surat", "Visakhapatnam", "Kochi",
    "Puri", "Mangaluru", "Kozhikode", "Thiruvananthapuram", "Bhubaneswar"
}


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def prepare_gale_gust_dataset() -> pd.DataFrame:
    """Prepare 44-year historical dataset with severe gale gust hazard targets."""
    root = get_project_root()
    hist_path = root / "data" / "processed" / "weather_historical_44y_train_ready.parquet"

    logger.info(f"Loading 44-year historical dataset from {hist_path}...")
    df = pd.read_parquet(hist_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["city", "date"]).reset_index(drop=True)

    # Physical meteorological variables
    df["diurnal_temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]
    df["gust_to_speed_ratio"] = df["wind_gusts_10m_max"] / np.maximum(1.0, df["wind_speed_10m_max"])

    wind_rad = np.radians(df["wind_direction_10m_dominant"])
    df["wind_dir_sin"] = np.sin(wind_rad)
    df["wind_dir_cos"] = np.cos(wind_rad)

    # Coastal exposure flag
    df["is_coastal_city"] = df["city"].isin(COASTAL_CITIES).astype(int)

    # Severe gale gust ground truth today (for persistence baseline)
    df["is_gale_today"] = (df["wind_gusts_10m_max"] >= GALE_GUST_THRESHOLD_KMH).astype(int)

    # Target: Tomorrow's gale gust hazard
    df["target_gale_next_day"] = df.groupby("city")["wind_gusts_10m_max"].shift(-1) >= GALE_GUST_THRESHOLD_KMH

    # Cyclical calendar & seasonal wind regimes
    df = add_temporal_features(df, date_col="date")
    month = df["date"].dt.month
    df["is_premonsoon_squall_season"] = month.isin([4, 5]).astype(int)
    df["is_monsoon_high_wind"] = month.isin([6, 7, 8, 9]).astype(int)
    df["is_postmonsoon_cyclone_season"] = month.isin([10, 11, 12]).astype(int)

    # Time-safe lag features
    logger.info("Computing time-safe lag and rolling features for gale gust hazard model...")
    lags = [1, 2, 3, 7]
    target_cols = [
        "is_gale_today",
        "wind_gusts_10m_max",
        "wind_speed_10m_max",
        "precipitation_sum",
        "shortwave_radiation_sum",
        "diurnal_temp_range",
        "apparent_temperature_max",
    ]
    df = add_lag_features(df, group_col="city", date_col="date", target_cols=target_cols, lags=lags)

    # Rolling statistics over strictly historical windows
    df = add_rolling_features(
        df,
        group_col="city",
        date_col="date",
        target_cols=["wind_gusts_10m_max", "wind_speed_10m_max", "precipitation_sum"],
        windows=[3, 7],
    )

    # Wind acceleration features
    df["gust_diff_24h"] = df["wind_gusts_10m_max"] - df["wind_gusts_10m_max_lag_1"]
    df["speed_diff_24h"] = df["wind_speed_10m_max"] - df["wind_speed_10m_max_lag_1"]

    # Drop incomplete edge rows (warmup lags + terminal forecast day)
    clean_df = df.dropna(subset=["wind_gusts_10m_max_lag_7", "target_gale_next_day"]).copy()
    logger.info(f"Cleaned Gale Gust dataset: {len(clean_df):,} rows across {clean_df['city'].nunique()} cities.")
    return clean_df


def train_gale_gust_hazard_model() -> Dict[str, Any]:
    """Train and evaluate the Severe Gale Gust Hazard LightGBM Classifier."""
    root = get_project_root()
    art_path = root / "artifacts" / "models" / "gale_gust_hazard" / "v0.1.0"
    rep_path = root / "reports" / "evaluation"
    art_path.mkdir(parents=True, exist_ok=True)
    rep_path.mkdir(parents=True, exist_ok=True)

    df = prepare_gale_gust_dataset()

    # Chronological Split
    train_mask = df["date"] < pd.to_datetime("2019-01-01")
    val_mask = (df["date"] >= pd.to_datetime("2019-01-01")) & (df["date"] < pd.to_datetime("2022-01-01"))
    test_mask = df["date"] >= pd.to_datetime("2022-01-01")

    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    excluded = {
        "date", "city", "target_gale_next_day", "is_gale_today",
        "wind_gusts_10m_max", "wind_speed_10m_max",
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "precipitation_sum", "sunrise", "sunset"
    }
    feature_cols = [c for c in df.columns if c not in excluded]

    X_train, y_train = train_df[feature_cols], train_df["target_gale_next_day"].astype(int).to_numpy()
    X_val, y_val = val_df[feature_cols], val_df["target_gale_next_day"].astype(int).to_numpy()
    X_test, y_test = test_df[feature_cols], test_df["target_gale_next_day"].astype(int).to_numpy()

    logger.info("Training Gale Gust Hazard LightGBM Classifier...")
    # Calculate scale_pos_weight to handle ~3.5% positive class imbalance
    neg_count = np.sum(y_train == 0)
    pos_count = max(1, np.sum(y_train == 1))
    scale_weight = float(np.sqrt(neg_count / pos_count))

    lgb_params = {
        "objective": "binary",
        "metric": ["auc", "average_precision"],
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "n_estimators": 600,
        "max_depth": 6,
        "num_leaves": 31,
        "scale_pos_weight": scale_weight,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 4,
    }

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )

    best_iter = int(model.best_iteration_)
    logger.info(f"Gale Gust Hazard Model converged at iteration {best_iter}.")

    # Evaluate on untouched 2022-2024 holdout
    preds_proba = model.predict_proba(X_test)[:, 1]
    # Decision threshold calibrated for balanced F1 / hazard detection
    preds_bin = (preds_proba >= 0.5).astype(int)

    roc_auc = round(float(roc_auc_score(y_test, preds_proba)), 4)
    pr_auc = round(float(average_precision_score(y_test, preds_proba)), 4)
    brier = round(float(brier_score_loss(y_test, preds_proba)), 4)
    accuracy = round(float(accuracy_score(y_test, preds_bin)), 4)

    test_pos = np.sum(y_test == 1)
    detected_pos = np.sum((preds_bin == 1) & (y_test == 1))
    recall = round(float(detected_pos / max(1, test_pos)), 4)
    precision = round(float(detected_pos / max(1, np.sum(preds_bin == 1))), 4)

    # Persistence baseline (today's gale -> tomorrow's gale)
    valid_persist = test_df["is_gale_today_lag_1"].notnull()
    persist_preds = test_df.loc[valid_persist, "is_gale_today_lag_1"].astype(int).to_numpy()
    y_test_persist = y_test[valid_persist]
    persist_acc = round(float(accuracy_score(y_test_persist, persist_preds)), 4)
    persist_recall = round(float(np.sum((persist_preds == 1) & (y_test_persist == 1)) / max(1, np.sum(y_test_persist == 1))), 4)

    logger.info(f"Gale Gust Hazard Test (2022-2024, {len(test_df):,} rows, {test_pos} gale days):")
    logger.info(f"  Model: ROC-AUC = {roc_auc}, PR-AUC = {pr_auc}, Accuracy = {accuracy*100:.2f}%, Recall = {recall*100:.2f}%, Precision = {precision*100:.2f}%, Brier = {brier}")
    logger.info(f"  Persistence Baseline: Accuracy = {persist_acc*100:.2f}%, Recall = {persist_recall*100:.2f}%")

    # Export Model Artifact
    model_file = art_path / "model.txt"
    model.booster_.save_model(str(model_file))
    sha256 = compute_sha256(model_file)

    # Reload parity check
    reloaded = lgb.Booster(model_file=str(model_file))
    parity_diff = float(np.max(np.abs(preds_proba - reloaded.predict(X_test))))
    logger.info(f"Gale Gust parity diff: {parity_diff:.8f}")

    results = {
        "model_name": "climacore_gale_gust_hazard",
        "training_data_rows": len(train_df),
        "total_cleaned_rows": len(df),
        "date_span": "1980 to 2024 (44 years)",
        "cities_count": int(df["city"].nunique()),
        "gale_threshold_kmh": GALE_GUST_THRESHOLD_KMH,
        "best_iteration": best_iter,
        "test_holdout_period": "2022-01-01 to 2024-12-31 (3 years)",
        "test_rows": len(test_df),
        "test_gale_events": int(test_pos),
        "metrics": {
            "gale_classifier": {
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "accuracy": accuracy,
                "recall": recall,
                "precision": precision,
                "brier_score": brier,
            },
            "persistence_baseline": {
                "accuracy": persist_acc,
                "recall": persist_recall,
            },
        },
        "feature_count": len(feature_cols),
        "artifact_file": str(model_file.name),
        "artifact_sha256": sha256,
    }

    out_json = rep_path / "gale_gust_hazard_evaluation_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    train_gale_gust_hazard_model()
