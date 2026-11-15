"""
Physical CPCB CAAQMS Station Model Training Pipeline.
Trained on 25,611 physical air quality monitoring station observations across 26 Indian cities,
integrating multi-pollutant chemistry (PM2.5, PM10, NO2, SO2, CO, O3) and meteorology
to forecast next-day ground-station PM2.5 concentrations and CPCB AQI hazard levels.
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import unicodedata
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

logger = get_logger("climacore.models.cpcb_station.train")


def normalize_city_name(name: str) -> str:
    """Normalize diacritics and common English variants for city matching."""
    nfkd = unicodedata.normalize("NFKD", str(name))
    clean = "".join([c for c in nfkd if not unicodedata.combining(c)])
    clean = clean.replace("Bangalore", "Bengaluru").replace("Vishakhapatnam", "Visakhapatnam")
    return clean


def prepare_cpcb_dataset() -> pd.DataFrame:
    """Load and process CPCB ground truth station data merged with atmospheric features."""
    root = get_project_root()
    cpcb_path = root / "data" / "processed" / "cpcb_ground_truth_cities.parquet"
    weather_path = root / "data" / "processed" / "weather_historical_44y_train_ready.parquet"

    logger.info(f"Loading CPCB CAAQMS ground truth observations from {cpcb_path}...")
    df_cpcb = pd.read_parquet(cpcb_path)
    df_cpcb["date"] = pd.to_datetime(df_cpcb["date"])
    df_cpcb["city_norm"] = df_cpcb["city"].apply(normalize_city_name)

    logger.info(f"Loading historical meteorology from {weather_path}...")
    df_w = pd.read_parquet(weather_path)
    df_w["date"] = pd.to_datetime(df_w["date"])
    df_w["city_norm"] = df_w["city"].apply(normalize_city_name)

    weather_cols = ["city_norm", "date", "temperature_2m_max", "precipitation_sum", "wind_speed_10m_max", "relative_humidity_2m_mean"]
    available_w_cols = [c for c in weather_cols if c in df_w.columns]
    merged = pd.merge(df_cpcb, df_w[available_w_cols], on=["city_norm", "date"], how="left")
    merged = merged.sort_values(by=["city", "date"]).reset_index(drop=True)

    # 1. Temporal Features
    merged = add_temporal_features(merged, date_col="date")
    merged["is_winter"] = merged["month"].isin([11, 12, 1, 2]).astype(int)
    merged["is_monsoon"] = merged["month"].isin([6, 7, 8, 9]).astype(int)

    # 2. Multi-Pollutant Lags
    lags = [1, 2, 3, 7]
    merged = add_lag_features(merged, group_col="city", date_col="date", target_cols=["cpcb_pm2_5_mean"], lags=lags)

    pollutant_cols = ["cpcb_no2_mean", "cpcb_so2_mean", "cpcb_co_mean", "cpcb_o3_mean"]
    merged = add_lag_features(merged, group_col="city", date_col="date", target_cols=pollutant_cols, lags=[1, 2])

    # 3. Rolling Statistics
    merged = add_rolling_features(merged, group_col="city", date_col="date", target_cols=["cpcb_pm2_5_mean"], windows=[3, 7])

    # 4. Atmospheric / Meteorological Lags
    met_cols = [c for c in ["wind_speed_10m_max", "precipitation_sum", "temperature_2m_max", "relative_humidity_2m_mean"] if c in merged.columns]
    if met_cols:
        merged = add_lag_features(merged, group_col="city", date_col="date", target_cols=met_cols, lags=[1])

    # 5. Target: Next-day CPCB station PM2.5 (t+1)
    merged["target_cpcb_pm2_5_next_day"] = merged.groupby("city")["cpcb_pm2_5_mean"].shift(-1)

    # Categorical city
    merged["city_cat"] = merged["city"].astype("category")

    # Drop warm-up rows and final target NaN
    valid_mask = merged["target_cpcb_pm2_5_next_day"].notnull() & merged["cpcb_pm2_5_mean_lag_1"].notnull()
    cleaned = merged[valid_mask].copy().reset_index(drop=True)
    logger.info(f"Cleaned CPCB station training dataset: {len(cleaned):,} rows across {cleaned['city'].nunique()} cities.")
    return cleaned


def train_cpcb_station_model() -> Dict[str, Any]:
    """Train and evaluate Physical CPCB CAAQMS Station Model."""
    root = get_project_root()
    art_path = root / "artifacts" / "models" / "cpcb_station" / "v0.1.0"
    rep_path = root / "reports" / "evaluation"
    art_path.mkdir(parents=True, exist_ok=True)
    rep_path.mkdir(parents=True, exist_ok=True)

    df = prepare_cpcb_dataset()

    # Chronological Split:
    # Train: 2015-01-01 to 2018-12-31 (4 years)
    # Val: 2019-01-01 to 2019-12-31 (1 year)
    # Test: 2020-01-01 to 2020-07-01 (untouched 2020 holdout)
    train_mask = df["date"] < pd.to_datetime("2019-01-01")
    val_mask = (df["date"] >= pd.to_datetime("2019-01-01")) & (df["date"] < pd.to_datetime("2020-01-01"))
    test_mask = df["date"] >= pd.to_datetime("2020-01-01")

    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    logger.info(
        f"Chronological Split: Train={len(train_df):,} rows ({train_df['date'].min():%Y-%m-%d} to {train_df['date'].max():%Y-%m-%d}), "
        f"Val={len(val_df):,} rows ({val_df['date'].min():%Y-%m-%d} to {val_df['date'].max():%Y-%m-%d}), "
        f"Test={len(test_df):,} rows ({test_df['date'].min():%Y-%m-%d} to {test_df['date'].max():%Y-%m-%d})"
    )

    excluded = {
        "date", "city", "city_norm", "target_cpcb_pm2_5_next_day",
        "cpcb_pm2_5_mean", "cpcb_pm10_mean", "cpcb_no2_mean", "cpcb_so2_mean", "cpcb_co_mean", "cpcb_o3_mean",
        "cpcb_aqi", "cpcb_aqi_category", "temperature_2m_max", "precipitation_sum", "wind_speed_10m_max", "relative_humidity_2m_mean"
    }
    feature_cols = [c for c in df.columns if c not in excluded]

    X_train, y_train = train_df[feature_cols], train_df["target_cpcb_pm2_5_next_day"].to_numpy()
    X_val, y_val = val_df[feature_cols], val_df["target_cpcb_pm2_5_next_day"].to_numpy()
    X_test, y_test = test_df[feature_cols], test_df["target_cpcb_pm2_5_next_day"].to_numpy()

    logger.info("Training CPCB Station LightGBM Regressor (RMSE objective)...")
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "n_estimators": 600,
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
    logger.info(f"CPCB Station Model converged at iteration {best_iter}.")

    # Evaluate on untouched 2020 holdout
    preds_test = np.maximum(0.0, model.predict(X_test))
    mae = round(float(mean_absolute_error(y_test, preds_test)), 4)
    rmse = round(float(root_mean_squared_error(y_test, preds_test)), 4)
    r2 = round(float(r2_score(y_test, preds_test)), 4)
    bias = round(float(np.mean(preds_test - y_test)), 4)

    # Persistence baseline (today's station PM2.5 -> tomorrow's station PM2.5)
    persist_mask = test_df["cpcb_pm2_5_mean_lag_1"].notnull()
    mae_persist = round(float(mean_absolute_error(y_test[persist_mask], test_df.loc[persist_mask, "cpcb_pm2_5_mean_lag_1"])), 4)
    rmse_persist = round(float(root_mean_squared_error(y_test[persist_mask], test_df.loc[persist_mask, "cpcb_pm2_5_mean_lag_1"])), 4)

    logger.info(f"CPCB Station Test (2020, {len(test_df):,} rows): MAE = {mae} µg/m³, RMSE = {rmse} µg/m³, R2 = {r2}")
    logger.info(f"CPCB Persistence Baseline: MAE = {mae_persist} µg/m³, RMSE = {rmse_persist} µg/m³")

    # Export Model Artifact
    model_file = art_path / "model.txt"
    model.booster_.save_model(str(model_file))
    sha256 = compute_sha256(model_file)

    # Reload parity check
    reloaded = lgb.Booster(model_file=str(model_file))
    reloaded_preds = np.maximum(0.0, reloaded.predict(X_test))
    parity_diff = float(np.max(np.abs(preds_test - reloaded_preds)))
    logger.info(f"CPCB Station parity diff: {parity_diff:.8f}")

    results = {
        "model_name": "climacore_cpcb_station",
        "training_data_rows": len(train_df),
        "total_cleaned_rows": len(df),
        "date_span": "2015 to 2020 (CPCB CAAQMS continuous ground monitoring network)",
        "cities_count": int(df["city"].nunique()),
        "best_iteration": best_iter,
        "test_holdout_period": "2020-01-01 to 2020-07-01",
        "test_rows": len(test_df),
        "metrics": {
            "cpcb_station_lgbm": {"mae": mae, "rmse": rmse, "r2": r2, "bias": bias},
            "persistence_baseline": {"mae": mae_persist, "rmse": rmse_persist},
        },
        "feature_count": len(feature_cols),
        "artifact_file": str(model_file.name),
        "artifact_sha256": sha256,
    }

    out_json = rep_path / "cpcb_station_evaluation_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    train_cpcb_station_model()
