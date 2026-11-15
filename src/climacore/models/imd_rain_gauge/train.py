"""
Physical IMD Rain Gauge Model Training Pipeline.
Trained on 5 years of IMD 0.25° gridded rain gauge observations (87,648 city-days)
merged with atmospheric reanalysis to predict next-day physical ground rain gauge accumulation.
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

logger = get_logger("climacore.models.imd_rain_gauge.train")


def prepare_imd_dataset() -> pd.DataFrame:
    """Load and merge IMD physical rainfall with reanalysis weather features."""
    root = get_project_root()
    imd_path = root / "data" / "processed" / "imd_rainfall_50_cities.parquet"
    weather_path = root / "data" / "raw" / "india_weather_air_quality" / "weather_daily.csv"

    logger.info(f"Loading IMD gauge observations from {imd_path}...")
    df_imd = pd.read_parquet(imd_path)

    logger.info(f"Loading weather reanalysis from {weather_path}...")
    df_w = pd.read_csv(weather_path)

    merged = pd.merge(df_imd, df_w, on=["city", "date"], how="inner")
    # Filter out missing coordinates / island stations without grid coverage
    merged = merged[merged["imd_rainfall_mm"].notnull()].copy()
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values(by=["city", "date"]).reset_index(drop=True)

    # 1. Temporal features
    merged = add_temporal_features(merged, date_col="date")
    merged["is_monsoon"] = merged["month"].isin([6, 7, 8, 9]).astype(int)
    merged["is_post_monsoon"] = merged["month"].isin([10, 11]).astype(int)

    # 2. Lags
    lags = [1, 2, 3, 7]
    target_cols = [
        "imd_rainfall_mm",
        "precipitation_sum",
        "relative_humidity_2m_mean",
        "wind_speed_10m_max",
        "temperature_2m_max",
        "temperature_2m_min",
    ]
    merged = add_lag_features(merged, group_col="city", date_col="date", target_cols=target_cols, lags=lags)

    # 3. Rolling statistics
    merged = add_rolling_features(merged, group_col="city", date_col="date", target_cols=["imd_rainfall_mm"], windows=[3, 7])
    merged = add_rolling_features(merged, group_col="city", date_col="date", target_cols=["precipitation_sum"], windows=[3, 7])

    # 4. Cross-domain interaction features
    merged["reanalysis_bias_lag_1"] = merged["precipitation_sum_lag_1"] - merged["imd_rainfall_mm_lag_1"]
    merged["both_rained_lag_1"] = (
        (merged["imd_rainfall_mm_lag_1"] >= 1.0) & (merged["precipitation_sum_lag_1"] >= 1.0)
    ).astype(int)

    # 5. Target: Next-day physical IMD rainfall (t+1)
    merged["target_imd_rain_next_day"] = merged.groupby("city")["imd_rainfall_mm"].shift(-1)

    # Categorical city
    merged["city_cat"] = merged["city"].astype("category")

    # Drop warm-up rows and final target NaN
    valid_mask = merged["target_imd_rain_next_day"].notnull() & merged["imd_rainfall_mm_lag_7"].notnull()
    cleaned = merged[valid_mask].copy().reset_index(drop=True)
    logger.info(f"Cleaned IMD gauge training dataset: {len(cleaned):,} rows across {cleaned['city'].nunique()} cities.")
    return cleaned


def train_imd_rain_gauge_model() -> Dict[str, Any]:
    """Train and evaluate Physical IMD Rain Gauge Model."""
    root = get_project_root()
    art_path = root / "artifacts" / "models" / "imd_rain_gauge" / "v0.1.0"
    rep_path = root / "reports" / "evaluation"
    art_path.mkdir(parents=True, exist_ok=True)
    rep_path.mkdir(parents=True, exist_ok=True)

    df = prepare_imd_dataset()

    # Chronological Split:
    # Train: 2021-01-08 to 2023-12-31 (3 years)
    # Val: 2024-01-01 to 2024-12-31 (1 year)
    # Test: 2025-01-01 to 2025-12-31 (1 year untouched holdout)
    train_mask = df["date"] < pd.to_datetime("2024-01-01")
    val_mask = (df["date"] >= pd.to_datetime("2024-01-01")) & (df["date"] < pd.to_datetime("2025-01-01"))
    test_mask = df["date"] >= pd.to_datetime("2025-01-01")

    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    logger.info(
        f"Chronological Split: Train={len(train_df):,} rows ({train_df['date'].min():%Y-%m-%d} to {train_df['date'].max():%Y-%m-%d}), "
        f"Val={len(val_df):,} rows ({val_df['date'].min():%Y-%m-%d} to {val_df['date'].max():%Y-%m-%d}), "
        f"Test={len(test_df):,} rows ({test_df['date'].min():%Y-%m-%d} to {test_df['date'].max():%Y-%m-%d})"
    )

    excluded = {"date", "city", "target_imd_rain_next_day", "imd_rainfall_mm"}
    feature_cols = [c for c in df.columns if c not in excluded]

    X_train, y_train = train_df[feature_cols], train_df["target_imd_rain_next_day"].to_numpy()
    X_val, y_val = val_df[feature_cols], val_df["target_imd_rain_next_day"].to_numpy()
    X_test, y_test = test_df[feature_cols], test_df["target_imd_rain_next_day"].to_numpy()

    # Model training with Huber loss (robust against extreme monsoon rainfall outliers)
    logger.info("Training IMD Rain Gauge LightGBM Regressor (Huber loss)...")
    lgb_params = {
        "objective": "huber",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "n_estimators": 800,
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
    logger.info(f"IMD Rain Gauge Model converged at iteration {best_iter}.")

    # Evaluate on untouched 2025 holdout
    preds_test = np.maximum(0.0, model.predict(X_test))
    mae = round(float(mean_absolute_error(y_test, preds_test)), 4)
    rmse = round(float(root_mean_squared_error(y_test, preds_test)), 4)
    r2 = round(float(r2_score(y_test, preds_test)), 4)
    bias = round(float(np.mean(preds_test - y_test)), 4)

    # Persistence baseline (today's IMD rain -> tomorrow's IMD rain)
    preds_persist = test_df["imd_rainfall_mm_lag_1"].to_numpy()
    mae_persist = round(float(mean_absolute_error(y_test, preds_persist)), 4)
    rmse_persist = round(float(root_mean_squared_error(y_test, preds_persist)), 4)

    logger.info(f"IMD Gauge Test (2025, {len(test_df):,} rows): MAE = {mae} mm, RMSE = {rmse} mm, R2 = {r2}")
    logger.info(f"IMD Persistence Baseline: MAE = {mae_persist} mm, RMSE = {rmse_persist} mm")

    # Export Model Artifact
    model_file = art_path / "model.txt"
    model.booster_.save_model(str(model_file))
    sha256 = compute_sha256(model_file)

    # Reload parity check
    reloaded = lgb.Booster(model_file=str(model_file))
    reloaded_preds = np.maximum(0.0, reloaded.predict(X_test))
    parity_diff = float(np.max(np.abs(preds_test - reloaded_preds)))
    logger.info(f"IMD Gauge parity diff: {parity_diff:.8f}")

    results = {
        "model_name": "climacore_imd_rain_gauge",
        "training_data_rows": len(train_df),
        "total_cleaned_rows": len(df),
        "date_span": "2021 to 2025 (5 years of IMD 0.25° gridded observations)",
        "cities_count": int(df["city"].nunique()),
        "best_iteration": best_iter,
        "test_holdout_period": "2025-01-01 to 2025-12-31 (1 full calendar year)",
        "test_rows": len(test_df),
        "metrics": {
            "imd_gauge_lgbm": {"mae": mae, "rmse": rmse, "r2": r2, "bias": bias},
            "persistence_baseline": {"mae": mae_persist, "rmse": rmse_persist},
        },
        "feature_count": len(feature_cols),
        "artifact_file": str(model_file.name),
        "artifact_sha256": sha256,
    }

    out_json = rep_path / "imd_rain_gauge_evaluation_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    train_imd_rain_gauge_model()
