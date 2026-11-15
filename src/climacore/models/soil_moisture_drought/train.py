"""Agricultural Drought & Root-Zone Soil Moisture Deficit Model.

Trains LightGBM models on 44 years of daily ERA5-Land multi-layer volumetric
soil moisture data (1980-2024, 820k+ records) across all 50 Indian cities.

Predicts:
1. Continuous next-day Root-Zone Soil Moisture (7-28 cm volumetric water content in m3/m3)
   and Soil Moisture Deficit Index (SMDI) for irrigation scheduling.
2. Binary Agricultural Flash Drought Hazard Alert (topsoil moisture <= 10th percentile).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict
import unicodedata

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)

from climacore.common.logging import get_logger
from climacore.common.provenance import compute_sha256
from climacore.config.loader import get_project_root
from climacore.features.lags import add_lag_features
from climacore.features.rolling import add_rolling_features
from climacore.features.temporal import add_temporal_features

logger = get_logger("climacore.models.soil_moisture_drought.train")


def _standardize_city(c: Any) -> str:
    norm = unicodedata.normalize("NFKD", str(c)).encode("ASCII", "ignore").decode("utf-8")
    if norm == "Bangalore":
        return "Bengaluru"
    return norm


def prepare_soil_moisture_dataset() -> pd.DataFrame:
    """Merge 44-year soil moisture with weather historical features."""
    root = get_project_root()
    soil_path = root / "data" / "processed" / "era5_soil_moisture_50_cities.parquet"
    weather_path = root / "data" / "processed" / "weather_historical_44y_train_ready.parquet"

    logger.info(f"Loading soil moisture dataset from {soil_path}...")
    soil_df = pd.read_parquet(soil_path)
    soil_df["date"] = pd.to_datetime(soil_df["date"])
    soil_df["city"] = soil_df["city"].apply(_standardize_city)

    logger.info(f"Loading 44-year weather reanalysis from {weather_path}...")
    weather_df = pd.read_parquet(weather_path)
    weather_df["date"] = pd.to_datetime(weather_df["date"])
    weather_df["city"] = weather_df["city"].apply(_standardize_city)

    # Selected weather drivers of evapotranspiration and soil drainage
    weather_subset = weather_df[[
        "city", "date", "temperature_2m_max", "temperature_2m_min",
        "precipitation_sum", "et0_fao_evapotranspiration",
        "shortwave_radiation_sum", "wind_speed_10m_max"
    ]].copy()

    df = pd.merge(soil_df, weather_subset, on=["city", "date"], how="inner")
    df = df.sort_values(by=["city", "date"]).reset_index(drop=True)

    # Physical water balance proxy: Precipitation - Reference Evapotranspiration
    df["daily_water_balance"] = (df["precipitation_sum"] - df["et0_fao_evapotranspiration"]).round(2)

    # Targets: Next-day root-zone moisture and flash drought status
    df["target_rootzone_moisture_next_day"] = df.groupby("city")["soil_moisture_7_to_28cm_mean"].shift(-1)
    df["target_topsoil_moisture_next_day"] = df.groupby("city")["soil_moisture_0_to_7cm_mean"].shift(-1)
    df["target_flash_drought_next_day"] = df.groupby("city")["is_flash_drought_day"].shift(-1)

    # Temporal & Seasonal features
    df = add_temporal_features(df, date_col="date")

    # Time-safe lag features
    logger.info("Computing time-safe lag features for soil moisture models...")
    lags = [1, 2, 3, 7, 14]
    lag_cols = [
        "soil_moisture_0_to_7cm_mean",
        "soil_moisture_7_to_28cm_mean",
        "soil_temperature_0_to_7cm_mean",
        "soil_moisture_deficit_index",
        "is_flash_drought_day",
        "precipitation_sum",
        "et0_fao_evapotranspiration",
        "daily_water_balance",
        "temperature_2m_max",
    ]
    df = add_lag_features(df, group_col="city", date_col="date", target_cols=lag_cols, lags=lags)

    # Rolling statistics over strictly historical windows
    df = add_rolling_features(
        df,
        group_col="city",
        date_col="date",
        target_cols=["soil_moisture_0_to_7cm_mean", "precipitation_sum", "et0_fao_evapotranspiration"],
        windows=[3, 7, 14],
    )

    # Soil moisture depletion rate over past 24 hours and 7 days
    df["topsoil_drying_rate_24h"] = (df["soil_moisture_0_to_7cm_mean_lag_1"] - df["soil_moisture_0_to_7cm_mean_lag_2"]).round(4)
    df["rootzone_drying_rate_7d"] = (df["soil_moisture_7_to_28cm_mean_lag_1"] - df["soil_moisture_7_to_28cm_mean_lag_7"]).round(4)

    # Drop edges
    clean_df = df.dropna(subset=["soil_moisture_0_to_7cm_mean_lag_14", "target_rootzone_moisture_next_day"]).copy()
    logger.info(f"Cleaned Soil Moisture & Drought dataset: {len(clean_df):,} rows across {clean_df['city'].nunique()} cities.")
    return clean_df


def train_soil_moisture_drought_models() -> Dict[str, Any]:
    """Train both Root-Zone Moisture Regressor and Flash Drought Classifier."""
    root = get_project_root()
    art_path = root / "artifacts" / "models" / "soil_moisture_drought" / "v0.1.0"
    rep_path = root / "reports" / "evaluation"
    art_path.mkdir(parents=True, exist_ok=True)
    rep_path.mkdir(parents=True, exist_ok=True)

    df = prepare_soil_moisture_dataset()

    # Chronological Split
    train_mask = df["date"] < pd.to_datetime("2019-01-01")
    val_mask = (df["date"] >= pd.to_datetime("2019-01-01")) & (df["date"] < pd.to_datetime("2022-01-01"))
    test_mask = df["date"] >= pd.to_datetime("2022-01-01")

    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    excluded = {
        "date", "city", "target_rootzone_moisture_next_day",
        "target_topsoil_moisture_next_day", "target_flash_drought_next_day",
        "soil_moisture_0_to_7cm_mean", "soil_moisture_7_to_28cm_mean",
        "soil_temperature_0_to_7cm_mean", "topsoil_moisture_anomaly",
        "rootzone_moisture_anomaly", "soil_moisture_deficit_index",
        "is_flash_drought_day", "precipitation_sum", "et0_fao_evapotranspiration",
        "temperature_2m_max", "temperature_2m_min", "daily_water_balance",
        "shortwave_radiation_sum", "wind_speed_10m_max"
    }
    feature_cols = [c for c in df.columns if c not in excluded]

    X_train, X_val, X_test = train_df[feature_cols], val_df[feature_cols], test_df[feature_cols]

    # ==========================================
    # Model 1: Root-Zone Soil Moisture Regressor
    # ==========================================
    y_reg_train = train_df["target_rootzone_moisture_next_day"].to_numpy()
    y_reg_val = val_df["target_rootzone_moisture_next_day"].to_numpy()
    y_reg_test = test_df["target_rootzone_moisture_next_day"].to_numpy()

    logger.info("Training Root-Zone Soil Moisture LightGBM Regressor...")
    reg_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "n_estimators": 600,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 4,
    }

    reg_model = lgb.LGBMRegressor(**reg_params)
    reg_model.fit(
        X_train,
        y_reg_train,
        eval_set=[(X_val, y_reg_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )

    reg_preds = reg_model.predict(X_test)
    reg_mae = round(float(mean_absolute_error(y_reg_test, reg_preds)), 5)
    reg_rmse = round(float(root_mean_squared_error(y_reg_test, reg_preds)), 5)
    reg_r2 = round(float(r2_score(y_reg_test, reg_preds)), 4)

    # Persistence baseline for root-zone moisture (lag-1 value)
    reg_persist = test_df["soil_moisture_7_to_28cm_mean_lag_1"].to_numpy()
    reg_persist_mae = round(float(mean_absolute_error(y_reg_test, reg_persist)), 5)
    reg_persist_rmse = round(float(root_mean_squared_error(y_reg_test, reg_persist)), 5)

    logger.info(f"Root-Zone Soil Moisture Test: MAE = {reg_mae} m3/m3, RMSE = {reg_rmse} m3/m3, R2 = {reg_r2}")
    logger.info(f"Persistence Baseline: MAE = {reg_persist_mae} m3/m3, RMSE = {reg_persist_rmse} m3/m3")

    # Export Regressor Artifact
    reg_file = art_path / "rootzone_model.txt"
    reg_model.booster_.save_model(str(reg_file))
    reg_sha256 = compute_sha256(reg_file)

    reloaded_reg = lgb.Booster(model_file=str(reg_file))
    reg_parity = float(np.max(np.abs(reg_preds - reloaded_reg.predict(X_test))))
    logger.info(f"Root-Zone reload parity: {reg_parity:.8f}")

    # ==========================================
    # Model 2: Agricultural Flash Drought Classifier
    # ==========================================
    y_cls_train = train_df["target_flash_drought_next_day"].astype(int).to_numpy()
    y_cls_val = val_df["target_flash_drought_next_day"].astype(int).to_numpy()
    y_cls_test = test_df["target_flash_drought_next_day"].astype(int).to_numpy()

    logger.info("Training Agricultural Flash Drought LightGBM Classifier...")
    cls_params = {
        "objective": "binary",
        "metric": ["auc", "average_precision"],
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "n_estimators": 600,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 4,
    }

    cls_model = lgb.LGBMClassifier(**cls_params)
    cls_model.fit(
        X_train,
        y_cls_train,
        eval_set=[(X_val, y_cls_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )

    cls_proba = cls_model.predict_proba(X_test)[:, 1]
    cls_bin = (cls_proba >= 0.5).astype(int)

    cls_roc_auc = round(float(roc_auc_score(y_cls_test, cls_proba)), 4)
    cls_pr_auc = round(float(average_precision_score(y_cls_test, cls_proba)), 4)
    cls_accuracy = round(float(accuracy_score(y_cls_test, cls_bin)), 4)
    cls_brier = round(float(brier_score_loss(y_cls_test, cls_proba)), 4)

    test_pos = np.sum(y_cls_test == 1)
    detected_pos = np.sum((cls_bin == 1) & (y_cls_test == 1))
    cls_recall = round(float(detected_pos / max(1, test_pos)), 4)
    cls_precision = round(float(detected_pos / max(1, np.sum(cls_bin == 1))), 4)

    # Persistence baseline for flash drought
    persist_cls = test_df["is_flash_drought_day_lag_1"].astype(int).to_numpy()
    persist_cls_acc = round(float(accuracy_score(y_cls_test, persist_cls)), 4)
    persist_cls_recall = round(float(np.sum((persist_cls == 1) & (y_cls_test == 1)) / max(1, test_pos)), 4)

    logger.info(f"Flash Drought Test: ROC-AUC = {cls_roc_auc}, PR-AUC = {cls_pr_auc}, Accuracy = {cls_accuracy*100:.2f}%, Recall = {cls_recall*100:.2f}%")
    logger.info(f"Persistence Baseline: Accuracy = {persist_cls_acc*100:.2f}%, Recall = {persist_cls_recall*100:.2f}%")

    cls_file = art_path / "flash_drought_model.txt"
    cls_model.booster_.save_model(str(cls_file))
    cls_sha256 = compute_sha256(cls_file)

    reloaded_cls = lgb.Booster(model_file=str(cls_file))
    cls_parity = float(np.max(np.abs(cls_proba - reloaded_cls.predict(X_test))))
    logger.info(f"Flash Drought reload parity: {cls_parity:.8f}")

    results = {
        "model_name": "climacore_soil_moisture_drought",
        "training_data_rows": len(train_df),
        "total_cleaned_rows": len(df),
        "date_span": "1980 to 2024 (44 years)",
        "cities_count": int(df["city"].nunique()),
        "test_holdout_period": "2022-01-01 to 2024-12-31 (3 years)",
        "test_rows": len(test_df),
        "metrics": {
            "rootzone_regressor": {
                "mae": reg_mae,
                "rmse": reg_rmse,
                "r2": reg_r2,
                "persistence_mae": reg_persist_mae,
                "persistence_rmse": reg_persist_rmse,
                "reload_parity_diff": reg_parity,
            },
            "flash_drought_classifier": {
                "roc_auc": cls_roc_auc,
                "pr_auc": cls_pr_auc,
                "accuracy": cls_accuracy,
                "recall": cls_recall,
                "precision": cls_precision,
                "brier_score": cls_brier,
                "persistence_accuracy": persist_cls_acc,
                "persistence_recall": persist_cls_recall,
                "reload_parity_diff": cls_parity,
            },
        },
        "feature_count": len(feature_cols),
        "artifacts": {
            "rootzone_model": {"file": str(reg_file.name), "sha256": reg_sha256},
            "flash_drought_model": {"file": str(cls_file.name), "sha256": cls_sha256},
        },
    }

    report_file = rep_path / "soil_moisture_drought_evaluation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    train_soil_moisture_drought_models()
