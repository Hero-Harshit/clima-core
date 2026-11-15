"""
Evaluation and demonstration runner for the CompoundRisk engine.
Fuses predictions from all three trained LightGBM hazard models over the
2026 chronological holdout test set and produces a multi-hazard intelligence report.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
import lightgbm as lgb
import numpy as np
import pandas as pd

from climacore.common.logging import get_logger
from climacore.config.loader import get_project_root
from climacore.models.compound_risk.config import HazardWeights
from climacore.models.compound_risk.scoring import CompoundRiskEngine

logger = get_logger("climacore.models.compound_risk.evaluate")


def run_compound_risk_evaluation() -> Dict[str, Any]:
    """Execute end-to-end multi-hazard compound risk evaluation."""
    root = get_project_root()
    art_dir = root / "artifacts" / "models"
    rep_dir = root / "reports" / "evaluation"
    rep_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load trained native LightGBM boosters
    logger.info("Loading trained hazard models...")
    aq_booster = lgb.Booster(model_file=str(art_dir / "air_quality" / "v0.1.0" / "model.txt"))
    heat_booster = lgb.Booster(model_file=str(art_dir / "heat_risk" / "v0.1.0" / "model.txt"))
    rain_clf = lgb.Booster(model_file=str(art_dir / "rain_risk" / "v0.1.0" / "classifier_model.txt"))
    rain_reg = lgb.Booster(model_file=str(art_dir / "rain_risk" / "v0.1.0" / "conditional_regressor_model.txt"))

    # 2. Load processed datasets
    logger.info("Loading processed datasets for 2026 test set evaluation...")
    df_aq = pd.read_parquet(root / "data" / "processed" / "air_quality_train_ready.parquet")
    df_heat = pd.read_parquet(root / "data" / "processed" / "heat_risk_train_ready.parquet")
    df_rain = pd.read_parquet(root / "data" / "processed" / "rain_risk_train_ready.parquet")

    # Filter to 2026 test period
    test_dt = pd.to_datetime("2026-01-01")
    df_aq["date"] = pd.to_datetime(df_aq["date"])
    df_heat["date"] = pd.to_datetime(df_heat["date"])
    df_rain["date"] = pd.to_datetime(df_rain["date"])

    test_aq = df_aq[df_aq["date"] >= test_dt].sort_values(by=["city", "date"]).reset_index(drop=True)
    test_heat = df_heat[df_heat["date"] >= test_dt].sort_values(by=["city", "date"]).reset_index(drop=True)
    test_rain = df_rain[df_rain["date"] >= test_dt].sort_values(by=["city", "date"]).reset_index(drop=True)

    # 3. Generate ML predictions independently on respective feature spaces
    logger.info("Generating predictions across all 3 hazard models...")
    test_aq["pred_pm2_5"] = aq_booster.predict(test_aq[aq_booster.feature_name()])
    test_heat["pred_temp_max"] = heat_booster.predict(test_heat[heat_booster.feature_name()])

    pred_rain_prob = rain_clf.predict(test_rain[rain_clf.feature_name()])
    pred_rain_cond = np.maximum(0.0, rain_reg.predict(test_rain[rain_reg.feature_name()]))
    test_rain["pred_precip"] = pred_rain_prob * pred_rain_cond
    test_rain["pred_rain_prob"] = pred_rain_prob

    # 4. Merge predictions on (city, date)
    merged = pd.merge(test_aq[["city", "date", "pred_pm2_5"]], test_heat[["city", "date", "pred_temp_max"]], on=["city", "date"], how="inner")
    merged = pd.merge(merged, test_rain[["city", "date", "pred_precip", "pred_rain_prob"]], on=["city", "date"], how="inner")
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")

    logger.info(f"Aligned multi-hazard predictions: {len(merged):,} records across 50 cities.")

    # 5. Run CompoundRiskEngine
    logger.info("Executing CompoundRiskEngine aggregation...")
    engine = CompoundRiskEngine()
    df_compound = engine.evaluate_dataframe(
        merged,
        pm2_5_col="pred_pm2_5",
        temp_max_col="pred_temp_max",
        precip_col="pred_precip",
        city_col="city",
        date_col="date",
    )

    # 6. Summary Statistics
    cat_counts = df_compound["severity_category"].value_counts().to_dict()
    compounded_events_count = int(df_compound["is_compounded_event"].sum())
    dominant_driver_counts = df_compound["dominant_driver"].value_counts().to_dict()

    # City Risk Ranking (Highest average compound risk in 2026)
    city_avg_risk = df_compound.groupby("city")["compound_risk_score"].mean().sort_values(ascending=False)
    top_5_cities = {k: round(float(v), 1) for k, v in city_avg_risk.head(5).items()}
    lowest_5_cities = {k: round(float(v), 1) for k, v in city_avg_risk.tail(5).items()}

    # High-profile Case Studies
    # A. Delhi in May (Pre-monsoon heatwave + dust/smog)
    delhi_may = df_compound[(df_compound["city"] == "Delhi") & (df_compound["date"].str.startswith("2026-05"))]
    delhi_peak = delhi_may.sort_values(by="compound_risk_score", ascending=False).iloc[0].to_dict() if len(delhi_may) > 0 else {}

    # B. Mumbai in July (Peak monsoon rainfall)
    mumbai_july = df_compound[(df_compound["city"] == "Mumbai") & (df_compound["date"].str.startswith("2026-07"))]
    mumbai_peak = mumbai_july.sort_values(by="compound_risk_score", ascending=False).iloc[0].to_dict() if len(mumbai_july) > 0 else {}

    report = {
        "engine": "Clima-Core CompoundRisk Engine",
        "version": "v0.1.0",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_test_observations": len(df_compound),
        "severity_distribution": cat_counts,
        "compounded_co_occurring_events": {
            "count": compounded_events_count,
            "percentage": round(float(compounded_events_count / len(df_compound) * 100), 2),
        },
        "dominant_hazard_drivers": dominant_driver_counts,
        "highest_risk_cities_avg_2026": top_5_cities,
        "lowest_risk_cities_avg_2026": lowest_5_cities,
        "case_studies": {
            "delhi_pre_monsoon_peak": delhi_peak,
            "mumbai_monsoon_peak": mumbai_peak,
        },
    }

    report_file = rep_dir / "compound_risk_evaluation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("CompoundRisk evaluation completed successfully. Report saved.")
    return report


if __name__ == "__main__":
    run_compound_risk_evaluation()
