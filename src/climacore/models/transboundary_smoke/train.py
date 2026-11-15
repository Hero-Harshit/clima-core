"""
Transboundary Smoke Drift & Wind Advection Model Training Pipeline.
Forecasts next-day cross-border aerosol and smoke transport across the Indo-Gangetic Plain
using wind vector alignment, seasonal biomass burning windows, and advection dynamics.
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import math
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, accuracy_score

from climacore.common.logging import get_logger
from climacore.common.provenance import compute_sha256
from climacore.config.loader import get_project_root
from climacore.features.temporal import add_temporal_features
from climacore.features.lags import add_lag_features
from climacore.features.rolling import add_rolling_features

logger = get_logger("climacore.models.transboundary_smoke.train")


def compute_advection_alignment(wind_dir_deg: pd.Series, wind_speed_kmh: pd.Series) -> pd.Series:
    """
    Computes Northwest-to-Southeast (NW-SE) Indo-Gangetic Plain wind vector alignment:
    Optimal trajectory is ~300° (WNW to ESE) with moderate transport velocity (10-25 km/h).
    """
    rad = np.radians(wind_dir_deg - 300.0)
    cos_proj = np.maximum(0.0, np.cos(rad))
    speed_factor = np.clip(wind_speed_kmh / 15.0, 0.2, 2.0)
    return (cos_proj * speed_factor).round(3)


def prepare_advection_dataset() -> pd.DataFrame:
    """Prepare 44-year historical dataset with Transboundary Advection ground truth."""
    root = get_project_root()
    hist_path = root / "data" / "processed" / "weather_historical_44y_train_ready.parquet"

    logger.info(f"Loading 44-year historical dataset from {hist_path}...")
    df = pd.read_parquet(hist_path)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    # Advection vector alignment score
    df["nw_advection_score"] = compute_advection_alignment(
        df["wind_direction_10m_dominant"].fillna(270.0),
        df["wind_speed_10m_max"].fillna(12.0)
    )

    # Active Stubble / Biomass Burning Seasons in Northern/Central India:
    # 1. Post-monsoon Kharif paddy stubble: Oct (10) and Nov (11)
    # 2. Pre-monsoon Rabi wheat stubble: Apr (4) and May (5)
    is_burning_season = df["month"].isin([10, 11, 4, 5])
    is_aligned = df["nw_advection_score"] >= 0.75
    is_transport_speed = (df["wind_speed_10m_max"] >= 10.0) & (df["wind_speed_10m_max"] <= 28.0)

    # Ground truth advection transport event
    df["is_advection_today"] = (is_burning_season & is_aligned & is_transport_speed).astype(int)

    # Feature Engineering
    logger.info("Computing time-safe lag and rolling features for smoke advection model...")
    df = df.sort_values(by=["city", "date"]).reset_index(drop=True)
    df = add_temporal_features(df, date_col="date")
    df["is_burning_season"] = is_burning_season.astype(int)

    lags = [1, 2, 3, 7]
    target_cols = [
        "is_advection_today",
        "nw_advection_score",
        "wind_speed_10m_max",
        "wind_direction_10m_dominant",
        "temperature_2m_max",
        "precipitation_sum",
    ]
    df = add_lag_features(df, group_col="city", date_col="date", target_cols=target_cols, lags=lags)

    df = add_rolling_features(df, group_col="city", date_col="date", target_cols=["nw_advection_score"], windows=[3, 7])
    df = add_rolling_features(df, group_col="city", date_col="date", target_cols=["wind_speed_10m_max"], windows=[3, 7])

    # Target: Next-day Transboundary Smoke Inflow (t+1)
    df["target_advection_next_day"] = df.groupby("city")["is_advection_today"].shift(-1)
    df["city_cat"] = df["city"].astype("category")

    valid_mask = df["target_advection_next_day"].notnull() & df["nw_advection_score_lag_7"].notnull()
    cleaned = df[valid_mask].copy().reset_index(drop=True)
    logger.info(f"Cleaned Smoke Advection dataset: {len(cleaned):,} rows across {cleaned['city'].nunique()} cities.")
    return cleaned


def train_transboundary_smoke_model() -> Dict[str, Any]:
    """Train LightGBM classifier for next-day transboundary smoke drift."""
    root = get_project_root()
    art_path = root / "artifacts" / "models" / "transboundary_smoke" / "v0.1.0"
    rep_path = root / "reports" / "evaluation"
    art_path.mkdir(parents=True, exist_ok=True)
    rep_path.mkdir(parents=True, exist_ok=True)

    df = prepare_advection_dataset()

    # Chronological Split
    train_mask = df["date"] < pd.to_datetime("2019-01-01")
    val_mask = (df["date"] >= pd.to_datetime("2019-01-01")) & (df["date"] < pd.to_datetime("2022-01-01"))
    test_mask = df["date"] >= pd.to_datetime("2022-01-01")

    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    excluded = {
        "date", "city", "target_advection_next_day", "is_advection_today",
        "nw_advection_score", "wind_direction_10m_dominant", "wind_speed_10m_max",
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "precipitation_sum", "sunrise", "sunset"
    }
    feature_cols = [c for c in df.columns if c not in excluded]

    X_train, y_train = train_df[feature_cols], train_df["target_advection_next_day"].astype(int).to_numpy()
    X_val, y_val = val_df[feature_cols], val_df["target_advection_next_day"].astype(int).to_numpy()
    X_test, y_test = test_df[feature_cols], test_df["target_advection_next_day"].astype(int).to_numpy()

    logger.info("Training Transboundary Smoke LightGBM Classifier...")
    lgb_params = {
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

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )

    best_iter = int(model.best_iteration_)
    logger.info(f"Transboundary Smoke Model converged at iteration {best_iter}.")

    # Evaluate on untouched 2022-2024 holdout
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds_bin = (preds_proba >= 0.5).astype(int)

    roc_auc = round(float(roc_auc_score(y_test, preds_proba)), 4)
    pr_auc = round(float(average_precision_score(y_test, preds_proba)), 4)
    brier = round(float(brier_score_loss(y_test, preds_proba)), 4)
    accuracy = round(float(accuracy_score(y_test, preds_bin)), 4)

    test_pos = np.sum(y_test == 1)
    detected_pos = np.sum((preds_bin == 1) & (y_test == 1))
    recall = round(float(detected_pos / max(1, test_pos)), 4)
    precision = round(float(detected_pos / max(1, np.sum(preds_bin == 1))), 4)

    # Persistence baseline (today's advection -> tomorrow's advection)
    valid_persist = test_df["is_advection_today_lag_1"].notnull()
    persist_preds = test_df.loc[valid_persist, "is_advection_today_lag_1"].astype(int).to_numpy()
    y_test_persist = y_test[valid_persist]
    persist_acc = round(float(accuracy_score(y_test_persist, persist_preds)), 4)
    persist_recall = round(float(np.sum((persist_preds == 1) & (y_test_persist == 1)) / max(1, np.sum(y_test_persist == 1))), 4)

    logger.info(f"Smoke Advection Test (2022-2024, {len(test_df):,} rows, {test_pos} advection days):")
    logger.info(f"  Model: ROC-AUC = {roc_auc}, PR-AUC = {pr_auc}, Accuracy = {accuracy*100:.2f}%, Recall = {recall*100:.2f}%, Precision = {precision*100:.2f}%, Brier = {brier}")
    logger.info(f"  Persistence Baseline: Accuracy = {persist_acc*100:.2f}%, Recall = {persist_recall*100:.2f}%")

    # Export Model Artifact
    model_file = art_path / "model.txt"
    model.booster_.save_model(str(model_file))
    sha256 = compute_sha256(model_file)

    # Reload parity check
    reloaded = lgb.Booster(model_file=str(model_file))
    parity_diff = float(np.max(np.abs(preds_proba - reloaded.predict(X_test))))
    logger.info(f"Smoke Advection parity diff: {parity_diff:.8f}")

    results = {
        "model_name": "climacore_transboundary_smoke",
        "training_data_rows": len(train_df),
        "total_cleaned_rows": len(df),
        "date_span": "1980 to 2024 (44 years)",
        "cities_count": int(df["city"].nunique()),
        "best_iteration": best_iter,
        "test_holdout_period": "2022-01-01 to 2024-12-31 (3 years)",
        "test_rows": len(test_df),
        "test_advection_events": int(test_pos),
        "metrics": {
            "advection_classifier": {
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

    out_json = rep_path / "transboundary_smoke_evaluation_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    train_transboundary_smoke_model()
