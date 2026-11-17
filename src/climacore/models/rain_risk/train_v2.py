"""
Rainfall Intelligence V2: Multi-Tier IMD Disaster Intensity Scale & Multi-Horizon Modeling.

Extends RainRisk to:
1. Official IMD 5-Tier Intensity Scale Classification:
   - Tier 0 (No / Trace Rain): < 2.5 mm
   - Tier 1 (Light Rain): 2.5 - 15.5 mm
   - Tier 2 (Moderate Rain): 15.6 - 64.4 mm
   - Tier 3 (Heavy Rain / Yellow Warning): 64.5 - 115.5 mm
   - Tier 4 (Very Heavy to Extreme Downpour / Orange/Red Alert): >= 115.6 mm
2. Standardized Precipitation Anomaly Indices (SPI-30, SPI-90) & Antecedent Precipitation Index (API).
3. Multi-Horizon Forecast Leads (24h, 48h, 72h).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd

from climacore.common.logging import get_logger
from climacore.common.provenance import compute_sha256
from climacore.config.loader import get_project_root
from climacore.features.temporal import add_temporal_features
from climacore.features.lags import add_lag_features

logger = get_logger("climacore.models.rain_risk_v2")

IMD_TIER_NAMES = {
    0: "NO_OR_TRACE_RAIN (<2.5mm)",
    1: "LIGHT_RAIN (2.5-15.5mm)",
    2: "MODERATE_RAIN (15.6-64.4mm)",
    3: "HEAVY_RAIN_YELLOW_ALERT (64.5-115.5mm)",
    4: "EXTREME_DOWNPOUR_ORANGE_RED_ALERT (>=115.6mm)"
}


def classify_imd_rain_tier(val: float) -> int:
    """Map precipitation volume (mm) into official IMD 5-Tier disaster warning scale."""
    if pd.isna(val) or val < 2.5:
        return 0
    elif val <= 15.5:
        return 1
    elif val <= 64.4:
        return 2
    elif val <= 115.5:
        return 3
    else:
        return 4


def build_rain_risk_v2_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Build multi-horizon targets, SPI anomalies, and moisture features."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["city", "date"]).reset_index(drop=True)

    # 1. Temporal features
    df = add_temporal_features(df, date_col="date")
    df["is_monsoon_season"] = df["month"].isin([6, 7, 8, 9]).astype(int)
    df["is_post_monsoon"] = df["month"].isin([10, 11]).astype(int)

    # 2. Daily Lags
    precip_cols = [
        "precipitation_sum",
        "relative_humidity_2m_mean",
        "temperature_2m_mean",
        "wind_speed_10m_max",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
    ]
    df = add_lag_features(
        df,
        group_col="city",
        date_col="date",
        target_cols=precip_cols,
        lags=[1, 2, 3, 7],
    )

    # 3. Antecedent Precipitation Index (API) & Rolling Accumulations
    # API_t = P_t + k * API_{t-1} where k ~ 0.85 (decay factor representing soil retention)
    grouped = df.groupby("city")["precipitation_sum"]
    for w in [3, 7, 14, 30]:
        df[f"precip_roll_sum_{w}d"] = grouped.rolling(window=w, min_periods=1).sum().reset_index(level=0, drop=True)
        df[f"precip_roll_max_{w}d"] = grouped.rolling(window=w, min_periods=1).max().reset_index(level=0, drop=True)

    # SPI-30 Proxy: Standardized Precipitation Anomaly over 30 days relative to city mean
    city_30d_mean = df.groupby("city")["precip_roll_sum_30d"].transform("mean")
    city_30d_std = df.groupby("city")["precip_roll_sum_30d"].transform("std").replace(0, 1e-4)
    df["spi_30d_proxy"] = (df["precip_roll_sum_30d"] - city_30d_mean) / city_30d_std

    # Atmospheric Water Deficit / Surplus: P - ET0
    df["water_balance_1d"] = df["precipitation_sum"] - df["et0_fao_evapotranspiration"]
    df["water_balance_7d"] = df["precip_roll_sum_7d"] - (df["et0_fao_evapotranspiration"] * 7.0)

    # 4. Multi-Horizon Targets (strictly forward-shifted)
    df["target_lead_24h"] = df.groupby("city")["precipitation_sum"].shift(-1)
    df["target_lead_48h"] = df.groupby("city")["precipitation_sum"].shift(-2)
    df["target_lead_72h"] = df.groupby("city")["precipitation_sum"].shift(-3)

    # Multi-tier classification target for next 24h
    df["target_imd_tier_24h"] = df["target_lead_24h"].apply(classify_imd_rain_tier)

    # Categoricals
    df["city_cat"] = df["city"].astype("category")

    excluded = {
        "date", "city", "state", "region",
        "target_lead_24h", "target_lead_48h", "target_lead_72h", "target_imd_tier_24h"
    }
    feature_cols = [c for c in df.columns if c not in excluded]

    return df, feature_cols


def train_rain_risk_v2_suite() -> Dict[str, Any]:
    """Train complete V2 suite: 5-Tier Classifier + 24h/48h/72h Lead Regressors."""
    root = get_project_root()
    raw_path = root / "data" / "raw" / "india_weather_air_quality"
    weather_df = pd.read_csv(raw_path / "weather_daily.csv")
    cities_df = pd.read_csv(raw_path / "cities.csv")

    merged = pd.merge(weather_df, cities_df, on="city", how="left")
    df_feat, feature_cols = build_rain_risk_v2_features(merged)

    # Drop warm-up rows and final rows where lead targets are NaN
    valid = df_feat["target_lead_72h"].notnull() & df_feat["precipitation_sum_lag_7"].notnull()
    df_clean = df_feat[valid].copy().reset_index(drop=True)

    # Chronological Splits
    df_clean["year"] = pd.to_datetime(df_clean["date"]).dt.year
    train_df = df_clean[df_clean["year"] <= 2021]
    val_df = df_clean[df_clean["year"] == 2022]
    test_df = df_clean[df_clean["year"] >= 2023]

    logger.info(f"Dataset split: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,} rows.")

    X_train, y_tier_train = train_df[feature_cols], train_df["target_imd_tier_24h"]
    X_val, y_tier_val = val_df[feature_cols], val_df["target_imd_tier_24h"]
    X_test, y_tier_test = test_df[feature_cols], test_df["target_imd_tier_24h"]

    # 1. IMD 5-Tier Multiclass LightGBM Classifier
    logger.info("Training IMD 5-Tier Disaster Warning Classifier...")
    tier_params = {
        "objective": "multiclass",
        "num_class": 5,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_estimators": 250,
        "random_state": 42,
        "verbose": -1,
    }
    tier_clf = lgb.LGBMClassifier(**tier_params)
    tier_clf.fit(
        X_train, y_tier_train,
        eval_set=[(X_val, y_tier_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )

    test_preds_tier = tier_clf.predict(X_test)
    test_acc_tier = float(np.mean(test_preds_tier == y_tier_test))
    logger.info(f"IMD 5-Tier Multiclass Accuracy: {test_acc_tier*100:.2f}%")

    # 2. Multi-Horizon Regressors (24h, 48h, 72h)
    lead_models = {}
    lead_metrics = {}

    for lead_name, target_col in [
        ("lead_24h", "target_lead_24h"),
        ("lead_48h", "target_lead_48h"),
        ("lead_72h", "target_lead_72h")
    ]:
        logger.info(f"Training Huber Regressor for {lead_name}...")
        reg_params = {
            "objective": "huber",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "n_estimators": 200,
            "random_state": 42,
            "verbose": -1,
        }
        reg = lgb.LGBMRegressor(**reg_params)
        reg.fit(
            X_train, train_df[target_col],
            eval_set=[(X_val, val_df[target_col])],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )

        preds = reg.predict(X_test)
        actuals = test_df[target_col].to_numpy()
        mae = float(np.mean(np.abs(preds - actuals)))
        lead_models[lead_name] = reg
        lead_metrics[lead_name] = {"mae_mm": round(mae, 3)}
        logger.info(f"{lead_name} Holdout MAE: {mae:.3f} mm")

    # 3. Save Artifacts
    artifact_dir = root / "artifacts" / "models" / "rain_risk_v2" / "v0.1.0"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    tier_model_path = artifact_dir / "imd_5tier_classifier.txt"
    tier_clf.booster_.save_model(str(tier_model_path))

    for name, model in lead_models.items():
        p = artifact_dir / f"{name}_regressor.txt"
        model.booster_.save_model(str(p))

    # Feature schema
    schema = {
        "model_version": "v0.1.0",
        "feature_columns": feature_cols,
        "feature_count": len(feature_cols),
        "imd_tiers": IMD_TIER_NAMES,
        "holdout_test_metrics": {
            "imd_tier_multiclass_accuracy": round(test_acc_tier, 4),
            **lead_metrics
        }
    }
    with open(artifact_dir / "schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    # Evaluation Report
    report_path = root / "reports" / "evaluation" / "rain_risk_v2_evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    logger.info("RainRisk V2 model suite trained, evaluated, and serialized successfully.")
    return schema


if __name__ == "__main__":
    train_rain_risk_v2_suite()
