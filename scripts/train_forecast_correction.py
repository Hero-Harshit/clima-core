"""Train and benchmark ForecastCorrectionEngine against independent IMD ground truth.

Evaluates Raw Baseline, Mean Bias Baseline, Empirical Quantile Mapping,
and LightGBM Residual Challenger on chronological holdout (train <= 2024-07-31, test > 2024-07-31).
Verifies against independent India Meteorological Department (IMD) observation products:
- imd_tmax_c (1.0 degree gridded temperature)
- imd_tmin_c (1.0 degree gridded temperature)
- imd_rainfall_mm (0.25 degree gridded rainfall)

Saves benchmark report to artifacts/models/forecast_correction/v0.1.0/.
"""

from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd

from climacore.analytics.forecast_correction import (
    ForecastCorrectionEngine,
    ForecastVariable,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DAILY_PAIRS_PATH = Path("data/processed/forecast_correction_pairs/daily_forecast_obs_pairs.parquet")


def main() -> None:
    if not DAILY_PAIRS_PATH.exists():
        raise FileNotFoundError(f"Paired forecast dataset not found at {DAILY_PAIRS_PATH}")

    df = pd.read_parquet(DAILY_PAIRS_PATH)
    logger.info("Loaded %d daily verified paired records.", len(df))

    engine = ForecastCorrectionEngine()

    weather_features = [
        "t2m_max_fcst",
        "t2m_min_fcst",
        "rh_mean_fcst",
        "ws_mean_fcst",
        "sin_doy",
        "cos_doy",
        "elevation",
        "latitude",
        "longitude",
    ]

    # 1. T2M_MAX vs IMD Ground Truth
    engine.train_and_benchmark(
        df=df,
        variable=ForecastVariable.T2M_MAX,
        fcst_col="t2m_max_fcst",
        obs_col="imd_tmax_c",
        feature_cols=weather_features,
        observation_source="IMD_Gridded_Obs_1deg",
    )

    # 2. T2M_MIN vs IMD Ground Truth
    engine.train_and_benchmark(
        df=df,
        variable=ForecastVariable.T2M_MIN,
        fcst_col="t2m_min_fcst",
        obs_col="imd_tmin_c",
        feature_cols=weather_features,
        observation_source="IMD_Gridded_Obs_1deg",
    )

    # 3. PRECIPITATION vs IMD Ground Truth (All-day + Conditional Wet-day)
    precip_features = [
        "precip_fcst",
        "rh_mean_fcst",
        "ws_mean_fcst",
        "sin_doy",
        "cos_doy",
        "elevation",
        "latitude",
        "longitude",
    ]
    engine.train_and_benchmark(
        df=df,
        variable=ForecastVariable.PRECIPITATION,
        fcst_col="precip_fcst",
        obs_col="imd_rainfall_mm",
        feature_cols=precip_features,
        observation_source="IMD_Gridded_Obs_0p25deg",
    )

    engine.save_state()

    print("\n" + "=" * 95)
    print("FORECAST CORRECTION CHRONOLOGICAL HOLDOUT BENCHMARK (VS INDEPENDENT IMD GROUND TRUTH)")
    print("=" * 95)
    for b in engine.benchmarks:
        print(
            f"[{b.variable:13s}] {b.method.value:17s} | Mask: {b.evaluation_mask:21s} | "
            f"MAE: {b.mae:6.3f} | RMSE: {b.rmse:6.3f} | MBE: {b.mbe:6.3f} | "
            f"r: {b.correlation_r:5.3f} | Skill: {b.skill_score:5.3f}"
        )
    print("=" * 95)

    print("\nPRECIPITATION OCCURRENCE & DETECTION CONTINGENCY:")
    for k, v in engine.contingency_metrics.items():
        print(f"[{k}] Acc: {v.accuracy:.3f} | Precision: {v.precision:.3f} | Recall: {v.recall:.3f} | F1: {v.f1_score:.3f} | Threat CSI: {v.threat_score_csi:.3f} | FAR: {v.false_alarm_ratio:.3f}")

    print("\nGOVERNANCE DECISIONS:")
    for k, v in engine.governance_decisions.items():
        print(f"  {k:15s} -> {v.value}")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
