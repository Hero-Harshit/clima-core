"""
Apparent Heat Stress & Heat Index Model Training Pipeline.
Trained on daily meteorological observations across 50 Indian cities,
integrating 2m dry-bulb temperature, relative humidity, solar radiation,
and wind speed to forecast next-day human thermal stress (apparent temperature / heat index).
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from climacore.common.logging import get_logger
from climacore.common.provenance import compute_sha256
from climacore.config.loader import get_project_root
from climacore.features.temporal import add_temporal_features
from climacore.features.lags import add_lag_features
from climacore.features.rolling import add_rolling_features

logger = get_logger("climacore.models.apparent_heat.train")


def prepare_apparent_heat_dataset() -> pd.DataFrame:
    """Load weather observations and compute bioclimatic heat stress features."""
    root = get_project_root()
    weather_path = root / "data" / "raw" / "india_weather_air_quality" / "weather_daily.csv"

    logger.info(f"Loading weather dataset from {weather_path}...")
    df = pd.read_csv(weather_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["city", "date"]).reset_index(drop=True)

    # 1. Temporal Features
    df = add_temporal_features(df, date_col="date")
    df["is_summer"] = df["month"].isin([3, 4, 5, 6]).astype(int)
    df["is_monsoon"] = df["month"].isin([7, 8, 9]).astype(int)

    # 2. Biophysical & Thermal Stress Lags
    # Note: Lags strictly use past information to prevent target leakage
    lags = [1, 2, 3, 7]
    thermal_cols = [
        "apparent_temperature_max",
        "temperature_2m_max",
        "temperature_2m_min",
        "relative_humidity_2m_mean",
        "shortwave_radiation_sum",
        "wind_speed_10m_max",
    ]
    df = add_lag_features(df, group_col="city", date_col="date", target_cols=thermal_cols, lags=lags)

    # 3. Rolling Thermal Stress History (heatwave accumulation)
    df = add_rolling_features(df, group_col="city", date_col="date", target_cols=["apparent_temperature_max"], windows=[3, 7])

    # 4. Thermal Excess feature: difference between apparent and dry-bulb temperature (sweat cooling impairment)
    df["thermal_excess_lag_1"] = df["apparent_temperature_max_lag_1"] - df["temperature_2m_max_lag_1"]

    # 5. Target: Next-day Apparent Temperature Max (t+1)
    df["target_apparent_heat_next_day"] = df.groupby("city")["apparent_temperature_max"].shift(-1)

    # Categorical city
    df["city_cat"] = df["city"].astype("category")

    # Drop warm-up rows (first 7 days) and final target NaN
    valid_mask = df["target_apparent_heat_next_day"].notnull() & df["apparent_temperature_max_lag_7"].notnull()
    cleaned = df[valid_mask].copy().reset_index(drop=True)
    logger.info(f"Cleaned apparent heat stress dataset: {len(cleaned):,} rows across {cleaned['city'].nunique()} cities.")
    return cleaned


def train_apparent_heat_model() -> Dict[str, Any]:
    """Train and evaluate Apparent Heat Stress & Heat Index LightGBM model."""
    root = get_project_root()
    art_path = root / "artifacts" / "models" / "apparent_heat_stress" / "v0.1.0"
    rep_path = root / "reports" / "evaluation"
    art_path.mkdir(parents=True, exist_ok=True)
    rep_path.mkdir(parents=True, exist_ok=True)

    df = prepare_apparent_heat_dataset()

    # Chronological Split:
    # Train: 2020-01-08 to 2024-12-31 (5 years, ~91,000 rows)
    # Val: 2025-01-01 to 2025-12-31 (1 year, ~18,000 rows)
    # Test: 2026-01-01 to 2026-10-30 (Holdout test, ~15,100 rows)
    train_mask = df["date"] < pd.to_datetime("2025-01-01")
    val_mask = (df["date"] >= pd.to_datetime("2025-01-01")) & (df["date"] < pd.to_datetime("2026-01-01"))
    test_mask = df["date"] >= pd.to_datetime("2026-01-01")

    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    logger.info(
        f"Chronological Split: Train={len(train_df):,} rows ({train_df['date'].min():%Y-%m-%d} to {train_df['date'].max():%Y-%m-%d}), "
        f"Val={len(val_df):,} rows ({val_df['date'].min():%Y-%m-%d} to {val_df['date'].max():%Y-%m-%d}), "
        f"Test={len(test_df):,} rows ({test_df['date'].min():%Y-%m-%d} to {test_df['date'].max():%Y-%m-%d})"
    )

    excluded = {
        "date", "city", "target_apparent_heat_next_day",
        "apparent_temperature_max", "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "relative_humidity_2m_mean", "shortwave_radiation_sum", "precipitation_sum", "precipitation_hours",
        "wind_speed_10m_max", "wind_direction_10m_dominant", "et0_fao_evapotranspiration", "weather_code"
    }
    feature_cols = [c for c in df.columns if c not in excluded]

    X_train, y_train = train_df[feature_cols], train_df["target_apparent_heat_next_day"].to_numpy()
    X_val, y_val = val_df[feature_cols], val_df["target_apparent_heat_next_day"].to_numpy()
    X_test, y_test = test_df[feature_cols], test_df["target_apparent_heat_next_day"].to_numpy()

    logger.info("Training Apparent Heat Stress LightGBM Regressor...")
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.04,
        "n_estimators": 700,
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
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )

    best_iter = int(model.best_iteration_)
    logger.info(f"Apparent Heat Stress Model converged at iteration {best_iter}.")

    # Evaluate on untouched 2026 holdout
    preds_test = model.predict(X_test)
    mae = round(float(mean_absolute_error(y_test, preds_test)), 4)
    rmse = round(float(root_mean_squared_error(y_test, preds_test)), 4)
    r2 = round(float(r2_score(y_test, preds_test)), 4)
    bias = round(float(np.mean(preds_test - y_test)), 4)

    # Persistence baseline (today's apparent temp -> tomorrow's apparent temp)
    preds_persist = test_df["apparent_temperature_max_lag_1"].to_numpy()
    mae_persist = round(float(mean_absolute_error(y_test, preds_persist)), 4)
    rmse_persist = round(float(root_mean_squared_error(y_test, preds_persist)), 4)

    # Extreme Thermal Stress Classification (Apparent Temp >= 41.0 °C - NOAA Danger category)
    high_heat_mask = y_test >= 41.0
    if np.sum(high_heat_mask) > 0:
        high_heat_pred_mask = preds_test >= 41.0
        high_heat_recall = round(float(np.sum(high_heat_pred_mask & high_heat_mask) / np.sum(high_heat_mask)), 4)
    else:
        high_heat_recall = 1.0

    logger.info(f"Apparent Heat Test (2026, {len(test_df):,} rows): MAE = {mae} °C, RMSE = {rmse} °C, R2 = {r2}")
    logger.info(f"Apparent Heat Persistence Baseline: MAE = {mae_persist} °C, RMSE = {rmse_persist} °C")
    logger.info(f"High Thermal Danger (>=41°C) Detection Recall: {high_heat_recall * 100:.2f}%")

    # Export Model Artifact
    model_file = art_path / "model.txt"
    model.booster_.save_model(str(model_file))
    sha256 = compute_sha256(model_file)

    # Reload parity check
    reloaded = lgb.Booster(model_file=str(model_file))
    parity_diff = float(np.max(np.abs(preds_test - reloaded.predict(X_test))))
    logger.info(f"Apparent Heat parity diff: {parity_diff:.8f}")

    results = {
        "model_name": "climacore_apparent_heat_stress",
        "training_data_rows": len(train_df),
        "total_cleaned_rows": len(df),
        "date_span": "2020 to 2026 (7 years)",
        "cities_count": int(df["city"].nunique()),
        "best_iteration": best_iter,
        "test_holdout_period": "2026-01-01 to 2026-10-30",
        "test_rows": len(test_df),
        "metrics": {
            "apparent_heat_lgbm": {"mae": mae, "rmse": rmse, "r2": r2, "bias": bias},
            "persistence_baseline": {"mae": mae_persist, "rmse": rmse_persist},
            "high_thermal_danger_recall_gte_41c": high_heat_recall,
        },
        "feature_count": len(feature_cols),
        "artifact_file": str(model_file.name),
        "artifact_sha256": sha256,
    }

    out_json = rep_path / "apparent_heat_evaluation_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    train_apparent_heat_model()
