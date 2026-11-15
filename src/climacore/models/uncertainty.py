"""
Quantile Uncertainty Regression Models for Clima-Core.
Trains LightGBM models with quantile loss (alpha=0.10, 0.50, 0.90) to produce
calibrated 80% prediction intervals for HeatRisk and AirQuality.
Evaluates empirical coverage on the 2026 chronological holdout.
"""

from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime, timezone
import lightgbm as lgb
import numpy as np
import pandas as pd

from climacore.common.logging import get_logger
from climacore.common.provenance import compute_sha256, get_file_metadata
from climacore.config.loader import get_project_root
from climacore.evaluation.splits import chronological_train_val_test_split

logger = get_logger("climacore.models.uncertainty")


def train_quantile_uncertainty_models(
    val_start: str = "2025-06-01",
    test_start: str = "2026-01-01",
) -> Dict[str, Any]:
    """Train alpha=0.10, 0.50, 0.90 quantile models for HeatRisk and AirQuality."""
    root = get_project_root()
    art_path = root / "artifacts" / "models" / "uncertainty" / "v0.1.0"
    rep_path = root / "reports" / "evaluation"
    art_path.mkdir(parents=True, exist_ok=True)
    rep_path.mkdir(parents=True, exist_ok=True)

    results = {
        "module": "quantile_uncertainty_regression",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "quantiles_trained": [0.10, 0.50, 0.90],
        "nominal_coverage_target": "80%",
    }

    # ==========================================
    # 1. HeatRisk Quantiles
    # ==========================================
    logger.info("Loading HeatRisk data for quantile regression...")
    df_heat = pd.read_parquet(root / "data" / "processed" / "heat_risk_train_ready.parquet")
    with open(root / "data" / "processed" / "heat_risk_feature_schema.json") as f:
        heat_feats = json.load(f)["feature_columns"]

    train_h, val_h, test_h = chronological_train_val_test_split(df_heat, "date", val_start="2025-01-01", test_start="2026-01-01")
    target_h = "target_temperature_2m_max_next_day"

    heat_quant_preds = {}
    for alpha in [0.10, 0.50, 0.90]:
        logger.info(f"Training HeatRisk Quantile alpha={alpha}...")
        q_params = {
            "objective": "quantile",
            "alpha": alpha,
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
        q_model = lgb.LGBMRegressor(**q_params)
        q_model.fit(
            train_h[heat_feats], train_h[target_h],
            eval_set=[(val_h[heat_feats], val_h[target_h])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        heat_quant_preds[alpha] = q_model.predict(test_h[heat_feats])
        q_file = art_path / f"heat_risk_quantile_{int(alpha*100)}.txt"
        q_model.booster_.save_model(str(q_file))

    # Calculate empirical coverage for HeatRisk
    y_test_h = test_h[target_h].to_numpy()
    in_interval_h = (y_test_h >= heat_quant_preds[0.10]) & (y_test_h <= heat_quant_preds[0.90])
    coverage_h = round(float(np.mean(in_interval_h) * 100), 2)
    avg_width_h = round(float(np.mean(heat_quant_preds[0.90] - heat_quant_preds[0.10])), 2)

    results["heat_risk"] = {
        "empirical_80pct_coverage": f"{coverage_h}%",
        "average_interval_width_deg_C": avg_width_h,
        "sample_prediction": {
            "p10_lower": round(float(heat_quant_preds[0.10][0]), 1),
            "p50_median": round(float(heat_quant_preds[0.50][0]), 1),
            "p90_upper": round(float(heat_quant_preds[0.90][0]), 1),
            "actual": round(float(y_test_h[0]), 1),
        }
    }
    logger.info(f"HeatRisk 80% interval empirical coverage: {coverage_h}% (avg width: {avg_width_h} °C)")

    # ==========================================
    # 2. AirQuality Quantiles
    # ==========================================
    logger.info("Loading AirQuality data for quantile regression...")
    df_aq = pd.read_parquet(root / "data" / "processed" / "air_quality_train_ready.parquet")
    with open(root / "data" / "processed" / "air_quality_feature_schema.json") as f:
        aq_feats = json.load(f)["feature_columns"]

    train_aq, val_aq, test_aq = chronological_train_val_test_split(df_aq, "date", val_start=val_start, test_start=test_start)
    target_aq = "target_pm2_5_next_day"

    aq_quant_preds = {}
    for alpha in [0.10, 0.50, 0.90]:
        logger.info(f"Training AirQuality Quantile alpha={alpha}...")
        q_params = {
            "objective": "quantile",
            "alpha": alpha,
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
        q_model = lgb.LGBMRegressor(**q_params)
        q_model.fit(
            train_aq[aq_feats], train_aq[target_aq],
            eval_set=[(val_aq[aq_feats], val_aq[target_aq])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        aq_quant_preds[alpha] = np.maximum(0.0, q_model.predict(test_aq[aq_feats]))
        q_file = art_path / f"air_quality_quantile_{int(alpha*100)}.txt"
        q_model.booster_.save_model(str(q_file))

    y_test_aq = test_aq[target_aq].to_numpy()
    in_interval_aq = (y_test_aq >= aq_quant_preds[0.10]) & (y_test_aq <= aq_quant_preds[0.90])
    coverage_aq = round(float(np.mean(in_interval_aq) * 100), 2)
    avg_width_aq = round(float(np.mean(aq_quant_preds[0.90] - aq_quant_preds[0.10])), 2)

    results["air_quality"] = {
        "empirical_80pct_coverage": f"{coverage_aq}%",
        "average_interval_width_ug_m3": avg_width_aq,
        "sample_prediction": {
            "p10_lower": round(float(aq_quant_preds[0.10][0]), 1),
            "p50_median": round(float(aq_quant_preds[0.50][0]), 1),
            "p90_upper": round(float(aq_quant_preds[0.90][0]), 1),
            "actual": round(float(y_test_aq[0]), 1),
        }
    }
    logger.info(f"AirQuality 80% interval empirical coverage: {coverage_aq}% (avg width: {avg_width_aq} ug/m3)")

    out_json = rep_path / "quantile_uncertainty_evaluation_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    train_quantile_uncertainty_models()
