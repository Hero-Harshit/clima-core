"""Train and evaluate FloodRisk SCS-CN Challenger Model.

Evaluates the LightGBM Flood Risk Champion against a new Challenger augmented
with deterministic USDA NRCS SCS Curve Number hydrological excess runoff features:
- scs_runoff_depth_mm (Q in mm)
- scs_runoff_coefficient (Q / P)
- scs_potential_retention_s (S in mm)

Evaluated on the identical chronological test holdout (test >= 2022-01-01).
Saves challenger booster and comparative evaluation report.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from climacore.config.loader import get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_FEATURES = [
    "rainfall_mm",
    "rain_rolling_3d",
    "rain_rolling_7d",
    "rain_rolling_14d",
    "rain_rolling_30d",
    "rain_max_3d",
    "api_14d",
    "precipitation_hours",
    "et0_fao_evapotranspiration",
    "wind_gusts_10m_max",
    "soil_moisture_0_to_7cm_mean",
    "soil_moisture_7_to_28cm_mean",
    "topsoil_moisture_anomaly",
    "rootzone_moisture_anomaly",
    "soil_moisture_deficit_index",
    "drainage_canals_15km",
    "bridges_15km",
    "transit_stations_15km",
    "population_15km",
    "dense_cluster_pop_15km",
]

SCS_FEATURES = [
    "scs_runoff_depth_mm",
    "scs_runoff_coefficient",
    "scs_potential_retention_s",
]

CHALLENGER_FEATURES = BASE_FEATURES + SCS_FEATURES
TARGET_COL = "flood_disaster_event"


def compute_scs_features(df: pd.DataFrame, default_cn2: float = 80.0) -> pd.DataFrame:
    """Compute vectorized deterministic SCS-CN features."""
    p = df["rainfall_mm"].values
    p5 = (df["rain_rolling_7d"].values * (5.0 / 7.0)).clip(min=0.0)

    cn2 = default_cn2
    cn1 = cn2 / (2.281 - 0.01281 * cn2)
    cn3 = cn2 / (0.427 + 0.00573 * cn2)

    # Dynamic AMC based on 5-day antecedent precipitation
    cn = np.where(p5 < 35.0, cn1, np.where(p5 > 52.5, cn3, cn2))
    s = (25400.0 / cn) - 254.0
    ia = 0.20 * s

    # Direct event runoff depth Q
    q = np.where(p > ia, ((p - ia) ** 2) / (p - ia + s), 0.0)
    # Runoff coefficient
    c = np.where(p > 0.0, np.divide(q, p, out=np.zeros_like(q), where=p > 0.0), 0.0)

    df_out = df.copy()
    df_out["scs_runoff_depth_mm"] = q
    df_out["scs_runoff_coefficient"] = c
    df_out["scs_potential_retention_s"] = s
    return df_out


def main() -> None:
    root = get_project_root()
    data_path = root / "data" / "processed" / "floods" / "city_flood_history_aligned.parquet"
    champ_model_path = root / "artifacts" / "models" / "flood_risk" / "v0.1.0" / "flood_risk_v1.booster"
    challenger_model_path = root / "artifacts" / "models" / "flood_risk" / "v0.1.0" / "flood_risk_challenger_scs_cn.booster"
    report_path = root / "reports" / "evaluation" / "flood_risk_scs_challenger_report.json"

    logger.info("Loading flood dataset from %s...", data_path)
    df = pd.read_parquet(data_path)
    df = compute_scs_features(df)

    train_mask = df["date"] <= "2018-12-31"
    val_mask = (df["date"] >= "2019-01-01") & (df["date"] <= "2021-12-31")
    test_mask = df["date"] >= "2022-01-01"

    y_train = df.loc[train_mask, TARGET_COL].values
    y_val = df.loc[val_mask, TARGET_COL].values
    y_test = df.loc[test_mask, TARGET_COL].values

    # 1. Evaluate Champion
    if not champ_model_path.exists():
        raise FileNotFoundError(f"Champion model not found at {champ_model_path}")
    champ_booster = lgb.Booster(model_file=str(champ_model_path))
    champ_test_preds = champ_booster.predict(df.loc[test_mask, BASE_FEATURES])

    champ_metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, champ_test_preds)), 4),
        "pr_auc": round(float(average_precision_score(y_test, champ_test_preds)), 4),
        "brier_score": round(float(brier_score_loss(y_test, champ_test_preds)), 4),
    }
    logger.info("Champion Metrics on Test Holdout: %s", champ_metrics)

    # 2. Train Challenger
    train_data = lgb.Dataset(df.loc[train_mask, CHALLENGER_FEATURES], label=y_train)
    val_data = lgb.Dataset(df.loc[val_mask, CHALLENGER_FEATURES], label=y_val, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "is_unbalance": True,
        "random_state": 42,
        "verbosity": -1,
    }

    logger.info("Training Challenger with SCS-CN Runoff Features...")
    challenger_booster = lgb.train(
        params,
        train_data,
        num_boost_round=120,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
    )

    challenger_booster.save_model(str(challenger_model_path))
    logger.info("Saved challenger booster to %s", challenger_model_path)

    chal_test_preds = challenger_booster.predict(df.loc[test_mask, CHALLENGER_FEATURES])
    chal_metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, chal_test_preds)), 4),
        "pr_auc": round(float(average_precision_score(y_test, chal_test_preds)), 4),
        "brier_score": round(float(brier_score_loss(y_test, chal_test_preds)), 4),
    }
    logger.info("Challenger Metrics on Test Holdout: %s", chal_metrics)

    # 3. Save comparative report
    report = {
        "evaluation_name": "FloodRisk_Champion_vs_SCS_CN_Challenger",
        "test_holdout": "date >= 2022-01-01",
        "test_sample_count": len(y_test),
        "champion": {
            "model_path": str(champ_model_path.relative_to(root)),
            "feature_count": len(BASE_FEATURES),
            "metrics": champ_metrics,
        },
        "challenger": {
            "model_path": str(challenger_model_path.relative_to(root)),
            "feature_count": len(CHALLENGER_FEATURES),
            "added_scs_features": SCS_FEATURES,
            "metrics": chal_metrics,
            "roc_auc_gain": round(chal_metrics["roc_auc"] - champ_metrics["roc_auc"], 4),
            "pr_auc_gain": round(chal_metrics["pr_auc"] - champ_metrics["pr_auc"], 4),
        },
        "disclaimer": (
            "SCS-CN runoff features represent hydrological excess rainfall depth and retention. "
            "Urban stormwater pipe backwater and drain overflow are not modeled."
        ),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved evaluation report to %s", report_path)


if __name__ == "__main__":
    main()
