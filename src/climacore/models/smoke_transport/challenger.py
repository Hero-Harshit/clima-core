"""Training and evaluation pipeline for SmokePath AirQuality PM2.5 Challenger.
Evaluates against existing v0.1.0 champion on identical 15,100 test samples (2026 chronological holdout).
"""

import os
import json
import hashlib
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from climacore.models.smoke_transport.engine import SmokePathTransportEngine


SMOKEPATH_FEATURES: List[str] = [
    "wind_u_component",
    "wind_v_component",
    "smoke_advection_index",
    "aod_smoke_coupling",
    "ventilation_index_proxy",
    "stagnation_inversion_index",
]

AQI_BREAKPOINTS = [0, 30, 60, 90, 120, 250, 5000]
AQI_CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def compute_aqi_category(pm25: np.ndarray) -> np.ndarray:
    return np.digitize(pm25, AQI_BREAKPOINTS[1:-1])


def evaluate_air_quality_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    r2 = float(r2_score(y_true, y_pred))
    
    true_cat = compute_aqi_category(y_true)
    pred_cat = compute_aqi_category(y_pred)
    acc = float(np.mean(true_cat == pred_cat))
    
    # Severe category recall (PM2.5 > 250 ug/m3, category index 5)
    severe_mask = true_cat == 5
    severe_recall = float(np.mean(pred_cat[severe_mask] == 5)) if np.sum(severe_mask) > 0 else 0.0
    
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mean_bias": round(bias, 4),
        "r2_score": round(r2, 4),
        "category_accuracy": round(acc, 4),
        "severe_category_recall": round(severe_recall, 4),
        "severe_events_count": int(np.sum(severe_mask)),
    }


def train_and_evaluate_smokepath():
    data_path = "data/processed/air_quality_train_ready.parquet"
    schema_path = "data/processed/air_quality_feature_schema.json"
    
    print("Loading air quality dataset...")
    df = pd.read_parquet(data_path)
    
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    base_features = schema["feature_columns"]
    target_col = schema["target_column"]
    
    print("Enriching dataset with SmokePath atmospheric transport features...")
    df = SmokePathTransportEngine.enrich_smoke_features(df)
    
    all_features = base_features + SMOKEPATH_FEATURES
    
    train_mask = df["date"] < "2025-06-01"
    val_mask = (df["date"] >= "2025-06-01") & (df["date"] < "2026-01-01")
    test_mask = df["date"] >= "2026-01-01"
    
    y_test = df.loc[test_mask, target_col].values
    print(f"Data splits: Train={train_mask.sum():,}, Val={val_mask.sum():,}, Test={test_mask.sum():,}")

    # 1. Evaluate Existing Champion (93 features)
    champ_model_path = "artifacts/models/air_quality/v0.1.0/model.txt"
    champ_booster = lgb.Booster(model_file=champ_model_path)
    champ_preds = champ_booster.predict(df.loc[test_mask, base_features])
    champ_metrics = evaluate_air_quality_predictions(y_test, champ_preds)
    print(f"\nAirQuality Champion (93 feats): MAE = {champ_metrics['mae']:.4f}, RMSE = {champ_metrics['rmse']:.4f}, R2 = {champ_metrics['r2_score']:.4f}, Severe Recall = {champ_metrics['severe_category_recall']:.4f}")

    # 2. Train SmokePath Challenger (99 features)
    print("\nTraining SmokePath PM2.5 Challenger (99 features)...")
    X_train = df.loc[train_mask, all_features]
    y_train = df.loc[train_mask, target_col].values
    X_val = df.loc[val_mask, all_features]
    y_val = df.loc[val_mask, target_col].values
    X_test = df.loc[test_mask, all_features]

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 450,
        "learning_rate": 0.05,
        "num_leaves": 45,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": 2,
        "verbose": -1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )

    challenger_preds = model.predict(X_test)
    challenger_metrics = evaluate_air_quality_predictions(y_test, challenger_preds)
    print(f"SmokePath Challenger (99 feats): MAE = {challenger_metrics['mae']:.4f}, RMSE = {challenger_metrics['rmse']:.4f}, R2 = {challenger_metrics['r2_score']:.4f}, Severe Recall = {challenger_metrics['severe_category_recall']:.4f}")

    # Feature importances
    feat_imps = dict(zip(all_features, [int(x) for x in model.feature_importances_]))
    smoke_imps = {k: feat_imps[k] for k in SMOKEPATH_FEATURES}
    print(f"\nSmokePath Feature Importances: {smoke_imps}")

    # Save artifact
    out_dir = "artifacts/models/v1_3/smoke_transport"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("reports/v1_3", exist_ok=True)
    booster_path = os.path.join(out_dir, "air_quality_smokepath_challenger.booster")
    model.booster_.save_model(booster_path)

    with open(booster_path, "rb") as f:
        sha256_hash = hashlib.sha256(f.read()).hexdigest()

    report = {
        "model_name": "SmokePath AirQuality PM2.5 Challenger V2",
        "features_count": len(all_features),
        "sha256": sha256_hash,
        "evaluation_period": "2026-01-01 to 2026-10-29",
        "test_samples": int(len(y_test)),
        "champion_v010": champ_metrics,
        "smokepath_challenger": challenger_metrics,
        "delta": {
            "mae_reduction": round(champ_metrics["mae"] - challenger_metrics["mae"], 4),
            "rmse_reduction": round(champ_metrics["rmse"] - challenger_metrics["rmse"], 4),
            "severe_recall_gain": round(challenger_metrics["severe_category_recall"] - champ_metrics["severe_category_recall"], 4),
        },
        "smokepath_feature_importances": smoke_imps,
        "scientific_disclaimer": (
            "Observational atmospheric advection and fire activity proxy. "
            "Does not constitute legal source attribution or chemical fingerprinting."
        ),
    }

    report_path = "reports/v1_3/smokepath_evaluation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved SmokePath evaluation report to {report_path}")

    return report


if __name__ == "__main__":
    train_and_evaluate_smokepath()
