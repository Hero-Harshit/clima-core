"""
CPCB Multi-Pollutant (PM10 and NO2) Model Training Pipeline.
Forecasts next-day physical ground monitoring station PM10 and NO2 concentrations
across Indian cities, completing the National Air Quality Index (NAQI) forecasting suite.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
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

logger = get_logger("climacore.models.cpcb_multipollutant.train")


def normalize_city_name(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(name))
    clean = "".join([c for c in nfkd if not unicodedata.combining(c)])
    clean = clean.replace("Bangalore", "Bengaluru").replace("Vishakhapatnam", "Visakhapatnam")
    return clean


def prepare_multipollutant_dataset() -> pd.DataFrame:
    """Load and process CPCB ground truth data with meteorology for PM10 and NO2."""
    root = get_project_root()
    cpcb_path = root / "data" / "processed" / "cpcb_ground_truth_cities.parquet"
    weather_path = root / "data" / "processed" / "weather_historical_44y_train_ready.parquet"

    logger.info(f"Loading CPCB ground truth observations from {cpcb_path}...")
    df_cpcb = pd.read_parquet(cpcb_path)
    df_cpcb["date"] = pd.to_datetime(df_cpcb["date"])
    df_cpcb["city_norm"] = df_cpcb["city"].apply(normalize_city_name)

    logger.info(f"Loading meteorology from {weather_path}...")
    df_w = pd.read_parquet(weather_path)
    df_w["date"] = pd.to_datetime(df_w["date"])
    df_w["city_norm"] = df_w["city"].apply(normalize_city_name)

    w_cols = ["city_norm", "date", "temperature_2m_max", "precipitation_sum", "wind_speed_10m_max"]
    avail_w = [c for c in w_cols if c in df_w.columns]
    merged = pd.merge(df_cpcb, df_w[avail_w], on=["city_norm", "date"], how="left")
    merged = merged.sort_values(by=["city", "date"]).reset_index(drop=True)

    # 1. Temporal Features
    merged = add_temporal_features(merged, date_col="date")
    merged["is_winter"] = merged["month"].isin([11, 12, 1, 2]).astype(int)
    merged["is_monsoon"] = merged["month"].isin([6, 7, 8, 9]).astype(int)

    # 2. Lags for PM10 and NO2
    lags = [1, 2, 3, 7]
    target_pollutants = ["cpcb_pm10_mean", "cpcb_no2_mean", "cpcb_pm2_5_mean"]
    merged = add_lag_features(merged, group_col="city", date_col="date", target_cols=target_pollutants, lags=lags)

    # 3. Rolling Statistics
    merged = add_rolling_features(merged, group_col="city", date_col="date", target_cols=["cpcb_pm10_mean"], windows=[3, 7])
    merged = add_rolling_features(merged, group_col="city", date_col="date", target_cols=["cpcb_no2_mean"], windows=[3, 7])

    # 4. Meteorology Lags
    met_cols = [c for c in ["wind_speed_10m_max", "precipitation_sum", "temperature_2m_max"] if c in merged.columns]
    if met_cols:
        merged = add_lag_features(merged, group_col="city", date_col="date", target_cols=met_cols, lags=[1])

    # 5. Targets (t+1)
    merged["target_pm10_next_day"] = merged.groupby("city")["cpcb_pm10_mean"].shift(-1)
    merged["target_no2_next_day"] = merged.groupby("city")["cpcb_no2_mean"].shift(-1)

    merged["city_cat"] = merged["city"].astype("category")
    return merged


def train_single_pollutant_model(
    df: pd.DataFrame,
    target_col: str,
    lag_col: str,
    pollutant_name: str,
    art_path: Path,
) -> Dict[str, Any]:
    """Train LightGBM regressor for a single pollutant."""
    valid_mask = df[target_col].notnull() & df[lag_col].notnull()
    df_clean = df[valid_mask].copy().reset_index(drop=True)

    train_mask = df_clean["date"] < pd.to_datetime("2019-01-01")
    val_mask = (df_clean["date"] >= pd.to_datetime("2019-01-01")) & (df_clean["date"] < pd.to_datetime("2020-01-01"))
    test_mask = df_clean["date"] >= pd.to_datetime("2020-01-01")

    train_df = df_clean[train_mask]
    val_df = df_clean[val_mask]
    test_df = df_clean[test_mask]

    excluded = {
        "date", "city", "city_norm", "target_pm10_next_day", "target_no2_next_day",
        "cpcb_pm2_5_mean", "cpcb_pm10_mean", "cpcb_no2_mean", "cpcb_so2_mean", "cpcb_co_mean", "cpcb_o3_mean",
        "cpcb_aqi", "cpcb_aqi_category", "temperature_2m_max", "precipitation_sum", "wind_speed_10m_max"
    }
    feature_cols = [c for c in df_clean.columns if c not in excluded]

    X_train, y_train = train_df[feature_cols], train_df[target_col].to_numpy()
    X_val, y_val = val_df[feature_cols], val_df[target_col].to_numpy()
    X_test, y_test = test_df[feature_cols], test_df[target_col].to_numpy()

    logger.info(f"Training {pollutant_name} LightGBM regressor ({len(train_df):,} train rows)...")
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
    preds_test = np.maximum(0.0, model.predict(X_test))
    mae = round(float(mean_absolute_error(y_test, preds_test)), 4)
    rmse = round(float(root_mean_squared_error(y_test, preds_test)), 4)
    r2 = round(float(r2_score(y_test, preds_test)), 4)
    bias = round(float(np.mean(preds_test - y_test)), 4)

    # Persistence baseline
    preds_persist = test_df[lag_col].to_numpy()
    mae_persist = round(float(mean_absolute_error(y_test, preds_persist)), 4)
    rmse_persist = round(float(root_mean_squared_error(y_test, preds_persist)), 4)

    logger.info(f"{pollutant_name} Test (2020 holdout, {len(test_df):,} rows): MAE = {mae} µg/m³, RMSE = {rmse} µg/m³, R2 = {r2}")
    logger.info(f"{pollutant_name} Persistence Baseline: MAE = {mae_persist} µg/m³, RMSE = {rmse_persist} µg/m³")

    # Export artifact
    model_file = art_path / f"{pollutant_name.lower()}_model.txt"
    model.booster_.save_model(str(model_file))
    sha256 = compute_sha256(model_file)

    # Parity check
    reloaded = lgb.Booster(model_file=str(model_file))
    reloaded_preds = np.maximum(0.0, reloaded.predict(X_test))
    parity_diff = float(np.max(np.abs(preds_test - reloaded_preds)))
    logger.info(f"{pollutant_name} parity diff: {parity_diff:.8f}")

    return {
        "pollutant": pollutant_name,
        "training_rows": len(train_df),
        "test_rows": len(test_df),
        "best_iteration": best_iter,
        "metrics": {
            "lgbm": {"mae": mae, "rmse": rmse, "r2": r2, "bias": bias},
            "persistence": {"mae": mae_persist, "rmse": rmse_persist},
        },
        "artifact_file": str(model_file.name),
        "artifact_sha256": sha256,
    }


def train_cpcb_multipollutant_models() -> Dict[str, Any]:
    """Train both PM10 and NO2 ground station forecasting models."""
    root = get_project_root()
    art_path = root / "artifacts" / "models" / "cpcb_multipollutant" / "v0.1.0"
    rep_path = root / "reports" / "evaluation"
    art_path.mkdir(parents=True, exist_ok=True)
    rep_path.mkdir(parents=True, exist_ok=True)

    df = prepare_multipollutant_dataset()

    res_pm10 = train_single_pollutant_model(
        df=df,
        target_col="target_pm10_next_day",
        lag_col="cpcb_pm10_mean_lag_1",
        pollutant_name="PM10",
        art_path=art_path,
    )

    res_no2 = train_single_pollutant_model(
        df=df,
        target_col="target_no2_next_day",
        lag_col="cpcb_no2_mean_lag_1",
        pollutant_name="NO2",
        art_path=art_path,
    )

    results = {
        "model_name": "climacore_cpcb_multipollutant",
        "date_span": "2015 to 2020",
        "cities_count": int(df["city"].nunique()),
        "test_holdout_period": "2020-01-01 to 2020-07-01",
        "pm10": res_pm10,
        "no2": res_no2,
    }

    out_json = rep_path / "cpcb_multipollutant_evaluation_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    train_cpcb_multipollutant_models()
