"""
Official IMD Heatwave Warning Classifier Training Pipeline.
Trained on 44 years of daily meteorological observations (823,048 records across 50 cities)
using official India Meteorological Department (IMD) climatological departure criteria
to forecast next-day heatwave warnings (Yellow / Orange / Red alerts).
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import unicodedata
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, classification_report

from climacore.common.logging import get_logger
from climacore.common.provenance import compute_sha256
from climacore.config.loader import get_project_root
from climacore.features.temporal import add_temporal_features
from climacore.features.lags import add_lag_features
from climacore.features.rolling import add_rolling_features

logger = get_logger("climacore.models.heatwave_alert.train")


def normalize_city_name(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(name))
    clean = "".join([c for c in nfkd if not unicodedata.combining(c)])
    clean = clean.replace("Bangalore", "Bengaluru").replace("Vishakhapatnam", "Visakhapatnam")
    return clean


def prepare_heatwave_dataset() -> pd.DataFrame:
    """Prepare 44-year historical dataset with IMD heatwave ground truth labels."""
    root = get_project_root()
    hist_path = root / "data" / "processed" / "weather_historical_44y_train_ready.parquet"
    cities_path = root / "data" / "raw" / "india_weather_air_quality" / "cities.csv"

    logger.info(f"Loading 44-year historical dataset from {hist_path}...")
    df = pd.read_parquet(hist_path)
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_year"] = df["date"].dt.day_of_year
    df["month"] = df["date"].dt.month
    df["city_norm"] = df["city"].apply(normalize_city_name)

    # 1. Compute 30-year WMO/IMD daily climate normals (1991-2020)
    logger.info("Computing official 30-year WMO climate normals (1991-2020)...")
    wmo_mask = (df["date"] >= "1991-01-01") & (df["date"] <= "2020-12-31")
    daily_normals = (
        df[wmo_mask]
        .groupby(["city_norm", "day_of_year"])["temperature_2m_max"]
        .mean()
        .rename("tmax_climatological_normal")
        .reset_index()
    )

    df = pd.merge(df, daily_normals, on=["city_norm", "day_of_year"], how="left")
    df["tmax_departure"] = df["temperature_2m_max"] - df["tmax_climatological_normal"]

    # 2. Integrate city elevation & geographical classifications
    df_cities = pd.read_csv(cities_path)
    df_cities["city_norm"] = df_cities["city"].apply(normalize_city_name)
    coastal_cities = {"Mumbai", "Chennai", "Kolkata", "Panaji", "Mangaluru", "Visakhapatnam", "Thiruvananthapuram", "Puducherry", "Kochi", "Surat"}

    df_cities["is_coastal"] = df_cities["city_norm"].isin(coastal_cities).astype(int)
    df_cities["is_hilly"] = (df_cities["elevation_m"] >= 1000.0).astype(int)

    df = pd.merge(df, df_cities[["city_norm", "elevation_m", "is_coastal", "is_hilly"]], on="city_norm", how="left")
    df["elevation_m"] = df["elevation_m"].fillna(200.0)
    df["is_coastal"] = df["is_coastal"].fillna(0).astype(int)
    df["is_hilly"] = df["is_hilly"].fillna(0).astype(int)

    # 3. Official IMD Heatwave Ground Truth Logic
    # Plains: Tmax >= 40 and departure >= 4.5 (or Tmax >= 45.0)
    # Coastal: Tmax >= 37 and departure >= 4.5
    # Hilly: Tmax >= 30 and departure >= 4.5
    is_plain = (df["is_coastal"] == 0) & (df["is_hilly"] == 0)
    plain_hw = is_plain & (((df["temperature_2m_max"] >= 40.0) & (df["tmax_departure"] >= 4.5)) | (df["temperature_2m_max"] >= 45.0))
    coastal_hw = (df["is_coastal"] == 1) & (df["temperature_2m_max"] >= 37.0) & (df["tmax_departure"] >= 4.5)
    hilly_hw = (df["is_hilly"] == 1) & (df["temperature_2m_max"] >= 30.0) & (df["tmax_departure"] >= 4.5)

    df["imd_heatwave_today"] = (plain_hw | coastal_hw | hilly_hw).astype(int)

    # Severe Heatwave: departure >= 6.5 or Tmax >= 47.0
    plain_shw = is_plain & (((df["temperature_2m_max"] >= 40.0) & (df["tmax_departure"] >= 6.5)) | (df["temperature_2m_max"] >= 47.0))
    coastal_shw = (df["is_coastal"] == 1) & (df["temperature_2m_max"] >= 37.0) & (df["tmax_departure"] >= 6.5)
    hilly_shw = (df["is_hilly"] == 1) & (df["temperature_2m_max"] >= 30.0) & (df["tmax_departure"] >= 6.5)
    df["imd_severe_heatwave_today"] = (plain_shw | coastal_shw | hilly_shw).astype(int)

    # 4. Feature Engineering
    logger.info("Computing time-safe lag and rolling features for heatwave prediction...")
    df = df.sort_values(by=["city", "date"]).reset_index(drop=True)
    df = add_temporal_features(df, date_col="date")
    df["is_summer"] = df["month"].isin([3, 4, 5, 6]).astype(int)

    lags = [1, 2, 3, 7]
    target_cols = [
        "temperature_2m_max",
        "tmax_departure",
        "imd_heatwave_today",
        "apparent_temperature_max",
        "shortwave_radiation_sum",
        "wind_speed_10m_max",
    ]
    df = add_lag_features(df, group_col="city", date_col="date", target_cols=target_cols, lags=lags)

    df = add_rolling_features(df, group_col="city", date_col="date", target_cols=["temperature_2m_max"], windows=[3, 7])
    df = add_rolling_features(df, group_col="city", date_col="date", target_cols=["tmax_departure"], windows=[3, 7])

    # Target: Next-day Heatwave Condition (t+1)
    df["target_heatwave_next_day"] = df.groupby("city")["imd_heatwave_today"].shift(-1)
    df["city_cat"] = df["city"].astype("category")

    valid_mask = df["target_heatwave_next_day"].notnull() & df["temperature_2m_max_lag_7"].notnull()
    cleaned = df[valid_mask].copy().reset_index(drop=True)
    logger.info(f"Cleaned Heatwave dataset: {len(cleaned):,} rows across {cleaned['city'].nunique()} cities.")
    return cleaned


def train_heatwave_alert_model() -> Dict[str, Any]:
    """Train LightGBM classifier for next-day official IMD heatwave warnings."""
    root = get_project_root()
    art_path = root / "artifacts" / "models" / "heatwave_alert" / "v0.1.0"
    rep_path = root / "reports" / "evaluation"
    art_path.mkdir(parents=True, exist_ok=True)
    rep_path.mkdir(parents=True, exist_ok=True)

    df = prepare_heatwave_dataset()

    # Chronological Split:
    # Train: 1980-01-08 to 2018-12-31 (~39 years, ~708,000 rows)
    # Val: 2019-01-01 to 2021-12-31 (3 years, ~54,800 rows)
    # Test: 2022-01-01 to 2024-12-31 (3 years, ~54,800 rows, including historic 2022 spring heatwave!)
    train_mask = df["date"] < pd.to_datetime("2019-01-01")
    val_mask = (df["date"] >= pd.to_datetime("2019-01-01")) & (df["date"] < pd.to_datetime("2022-01-01"))
    test_mask = df["date"] >= pd.to_datetime("2022-01-01")

    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    excluded = {
        "date", "city", "city_norm", "target_heatwave_next_day",
        "imd_heatwave_today", "imd_severe_heatwave_today", "temperature_2m_max", "tmax_departure",
        "sunrise", "sunset"
    }
    feature_cols = [c for c in df.columns if c not in excluded]

    X_train, y_train = train_df[feature_cols], train_df["target_heatwave_next_day"].astype(int).to_numpy()
    X_val, y_val = val_df[feature_cols], val_df["target_heatwave_next_day"].astype(int).to_numpy()
    X_test, y_test = test_df[feature_cols], test_df["target_heatwave_next_day"].astype(int).to_numpy()

    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    scale_pos = round(float(neg_count / max(1, pos_count)), 2)
    logger.info(f"Class distribution: Positives={pos_count:,}, Negatives={neg_count:,}, scale_pos_weight={scale_pos}")

    # Cost-sensitive LightGBM classifier
    lgb_params = {
        "objective": "binary",
        "metric": ["auc", "average_precision"],
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "n_estimators": 600,
        "max_depth": 6,
        "num_leaves": 31,
        "scale_pos_weight": min(scale_pos, 15.0),  # balanced cost sensitivity
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
    logger.info(f"Heatwave Alert Model converged at iteration {best_iter}.")

    # Evaluate on untouched 2022-2024 holdout
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds_bin = (preds_proba >= 0.5).astype(int)

    roc_auc = round(float(roc_auc_score(y_test, preds_proba)), 4)
    pr_auc = round(float(average_precision_score(y_test, preds_proba)), 4)
    brier = round(float(brier_score_loss(y_test, preds_proba)), 4)

    test_pos = np.sum(y_test == 1)
    detected_pos = np.sum((preds_bin == 1) & (y_test == 1))
    recall = round(float(detected_pos / max(1, test_pos)), 4)
    precision = round(float(detected_pos / max(1, np.sum(preds_bin == 1))), 4)

    # Persistence baseline (today was heatwave -> tomorrow will be heatwave)
    persist_preds = test_df["imd_heatwave_today_lag_1"].astype(int).to_numpy()
    persist_recall = round(float(np.sum((persist_preds == 1) & (y_test == 1)) / max(1, test_pos)), 4)
    persist_precision = round(float(np.sum((persist_preds == 1) & (y_test == 1)) / max(1, np.sum(persist_preds == 1))), 4)

    logger.info(f"IMD Heatwave Test (2022-2024, {len(test_df):,} rows, {test_pos} heatwave days):")
    logger.info(f"  Model: ROC-AUC = {roc_auc}, PR-AUC = {pr_auc}, Recall = {recall * 100:.1f}%, Precision = {precision * 100:.1f}%, Brier = {brier}")
    logger.info(f"  Persistence Baseline: Recall = {persist_recall * 100:.1f}%, Precision = {persist_precision * 100:.1f}%")

    # Export Model Artifact
    model_file = art_path / "model.txt"
    model.booster_.save_model(str(model_file))
    sha256 = compute_sha256(model_file)

    # Reload parity check
    reloaded = lgb.Booster(model_file=str(model_file))
    parity_diff = float(np.max(np.abs(preds_proba - reloaded.predict(X_test))))
    logger.info(f"Heatwave Alert parity diff: {parity_diff:.8f}")

    results = {
        "model_name": "climacore_imd_heatwave_alert",
        "training_data_rows": len(train_df),
        "total_cleaned_rows": len(df),
        "date_span": "1980 to 2024 (44 years)",
        "cities_count": int(df["city"].nunique()),
        "best_iteration": best_iter,
        "test_holdout_period": "2022-01-01 to 2024-12-31 (3 years)",
        "test_rows": len(test_df),
        "test_heatwave_events": int(test_pos),
        "metrics": {
            "heatwave_classifier": {
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "recall": recall,
                "precision": precision,
                "brier_score": brier,
            },
            "persistence_baseline": {
                "recall": persist_recall,
                "precision": persist_precision,
            },
        },
        "feature_count": len(feature_cols),
        "artifact_file": str(model_file.name),
        "artifact_sha256": sha256,
    }

    out_json = rep_path / "heatwave_alert_evaluation_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    train_heatwave_alert_model()
