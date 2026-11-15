"""
Training pipeline for the 44-Year Climate Master Model (1980-2024).
Trained across 823,048 daily records covering 4.5 decades of Indian climate,
historic heatwaves, and seasonal shifts across 50 cities.
"""

from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime, timezone
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from climacore.common.logging import get_logger
from climacore.common.provenance import compute_sha256, get_file_metadata
from climacore.config.loader import get_project_root
from climacore.features.temporal import add_temporal_features
from climacore.features.lags import add_lag_features
from climacore.features.rolling import add_rolling_features

logger = get_logger("climacore.models.climate_master.train")


def train_climate_master_model() -> Dict[str, Any]:
    """Train LightGBM regressor on 44 years of daily weather data."""
    root = get_project_root()
    art_path = root / "artifacts" / "models" / "climate_master_44y" / "v0.1.0"
    rep_path = root / "reports" / "evaluation"
    art_path.mkdir(parents=True, exist_ok=True)
    rep_path.mkdir(parents=True, exist_ok=True)

    parquet_file = root / "data" / "processed" / "weather_historical_44y_train_ready.parquet"
    logger.info(f"Loading 44-year historical weather dataset from {parquet_file}...")
    df = pd.read_parquet(parquet_file)

    # 1. Feature Engineering
    logger.info("Computing time-safe features for 44-year dataset...")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["city", "date"]).reset_index(drop=True)

    df = add_temporal_features(df, date_col="date")

    # Lags
    lags = [1, 2, 3, 7]
    target_cols = [
        "temperature_2m_max",
        "temperature_2m_min",
        "apparent_temperature_max",
        "precipitation_sum",
        "wind_speed_10m_max",
        "shortwave_radiation_sum",
    ]
    df = add_lag_features(df, group_col="city", date_col="date", target_cols=target_cols, lags=lags)

    # Rolling statistics
    df = add_rolling_features(df, group_col="city", date_col="date", target_cols=["temperature_2m_max"], windows=[3, 7, 14])

    # Target: Next-day maximum temperature (t+1)
    df["target_tmax_next_day"] = df.groupby("city")["temperature_2m_max"].shift(-1)

    # Convert city to category
    df["city_cat"] = df["city"].astype("category")

    # Drop warm-up rows (first 7 days) and terminal target NaN
    valid_mask = df["target_tmax_next_day"].notnull() & df["temperature_2m_max_lag_7"].notnull()
    df_clean = df[valid_mask].copy().reset_index(drop=True)
    logger.info(f"Cleaned 44-year training dataset: {len(df_clean):,} rows.")

    # 2. Chronological Split
    # Train: 1980-01-08 to 2018-12-31 (~39 years, ~710,000 rows)
    # Val: 2019-01-01 to 2021-12-31 (3 years, ~55,000 rows)
    # Test: 2022-01-01 to 2024-12-31 (3 years, ~55,000 rows untouched)
    train_mask = df_clean["date"] < pd.to_datetime("2019-01-01")
    val_mask = (df_clean["date"] >= pd.to_datetime("2019-01-01")) & (df_clean["date"] < pd.to_datetime("2022-01-01"))
    test_mask = (df_clean["date"] >= pd.to_datetime("2022-01-01")) & (df_clean["date"] <= pd.to_datetime("2024-12-31"))

    train_df = df_clean[train_mask]
    val_df = df_clean[val_mask]
    test_df = df_clean[test_mask]

    logger.info(
        f"Chronological 44-Year Split: "
        f"Train={len(train_df):,} rows ({train_df['date'].min():%Y-%m-%d} to {train_df['date'].max():%Y-%m-%d}), "
        f"Val={len(val_df):,} rows ({val_df['date'].min():%Y-%m-%d} to {val_df['date'].max():%Y-%m-%d}), "
        f"Test={len(test_df):,} rows ({test_df['date'].min():%Y-%m-%d} to {test_df['date'].max():%Y-%m-%d})"
    )

    excluded = {"date", "city", "target_tmax_next_day", "sunrise", "sunset"}
    feature_cols = [c for c in df_clean.columns if c not in excluded]

    X_train, y_train = train_df[feature_cols], train_df["target_tmax_next_day"].to_numpy()
    X_val, y_val = val_df[feature_cols], val_df["target_tmax_next_day"].to_numpy()
    X_test, y_test = test_df[feature_cols], test_df["target_tmax_next_day"].to_numpy()

    # 3. Train LightGBM Model
    logger.info("Training 44-Year Climate Master LightGBM model...")
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 4,
    }

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    best_iter = int(model.best_iteration_)
    logger.info(f"Climate Master Model converged at iteration {best_iter}.")

    # 4. Evaluate on 3-Year Untouched Test Set (2022-2024)
    preds_test = model.predict(X_test)
    mae = round(float(mean_absolute_error(y_test, preds_test)), 4)
    rmse = round(float(root_mean_squared_error(y_test, preds_test)), 4)
    r2 = round(float(r2_score(y_test, preds_test)), 4)
    bias = round(float(np.mean(preds_test - y_test)), 4)

    # Persistence Baseline
    preds_persist = test_df["temperature_2m_max"].to_numpy()
    mae_persist = round(float(mean_absolute_error(y_test, preds_persist)), 4)
    rmse_persist = round(float(root_mean_squared_error(y_test, preds_persist)), 4)

    logger.info(f"Climate Master Test (2022-2024, {len(test_df):,} rows): MAE = {mae} °C, RMSE = {rmse} °C, R2 = {r2}")
    logger.info(f"Persistence Baseline: MAE = {mae_persist} °C, RMSE = {rmse_persist} °C")

    # 5. Export Model Artifact
    model_file = art_path / "model.txt"
    model.booster_.save_model(str(model_file))
    sha256 = compute_sha256(model_file)

    # Reload parity check
    reloaded = lgb.Booster(model_file=str(model_file))
    parity_diff = np.max(np.abs(preds_test - reloaded.predict(X_test)))
    logger.info(f"Climate Master parity diff: {parity_diff:.8f}")

    results = {
        "model_name": "climacore_climate_master_44y",
        "training_data_rows": len(train_df),
        "total_historical_rows": len(df_clean),
        "date_span": "1980 to 2024 (44 years)",
        "cities_count": int(df_clean["city"].nunique()),
        "best_iteration": best_iter,
        "test_holdout_period": "2022-01-01 to 2024-12-31 (3 years)",
        "test_rows": len(test_df),
        "metrics": {
            "climate_master_lgbm": {"mae": mae, "rmse": rmse, "r2": r2, "bias": bias},
            "persistence_baseline": {"mae": mae_persist, "rmse": rmse_persist},
        },
        "artifact_file": str(model_file.name),
        "artifact_sha256": sha256,
    }

    out_json = rep_path / "climate_master_evaluation_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    train_climate_master_model()
