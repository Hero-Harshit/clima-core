"""Train and benchmark ForecastCorrectionEngine on paired forecast-observation data.

Evaluates Raw Baseline, Mean Bias Baseline, Empirical Quantile Mapping,
and LightGBM Residual Challenger on chronological holdout (train <= 2024-07-31, test > 2024-07-31).
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
    logger.info("Loaded %d daily paired records.", len(df))

    engine = ForecastCorrectionEngine()

    # Features for residual learning
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

    # 1. T2M_MAX
    m_max = engine.train_and_benchmark(
        df=df,
        variable=ForecastVariable.T2M_MAX,
        fcst_col="t2m_max_fcst",
        obs_col="t2m_max_obs",
        feature_cols=weather_features,
    )

    # 2. T2M_MIN
    m_min = engine.train_and_benchmark(
        df=df,
        variable=ForecastVariable.T2M_MIN,
        fcst_col="t2m_min_fcst",
        obs_col="t2m_min_obs",
        feature_cols=weather_features,
    )

    # 3. PRECIPITATION
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
    m_precip = engine.train_and_benchmark(
        df=df,
        variable=ForecastVariable.PRECIPITATION,
        fcst_col="precip_fcst",
        obs_col="precip_obs",
        feature_cols=precip_features,
    )

    # 4. WIND SPEED
    m_ws = engine.train_and_benchmark(
        df=df,
        variable=ForecastVariable.WIND_SPEED,
        fcst_col="ws_mean_fcst",
        obs_col="ws_mean_obs",
        feature_cols=weather_features,
    )

    engine.save_state()

    print("\n" + "=" * 80)
    print("FORECAST CORRECTION CHRONOLOGICAL HOLDOUT BENCHMARK REPORT")
    print("=" * 80)
    for b in engine.benchmarks:
        print(
            f"[{b.variable:14s}] {b.method.value:18s} | "
            f"MAE: {b.mae:7.3f} | RMSE: {b.rmse:7.3f} | MBE: {b.mbe:7.3f} | "
            f"r: {b.correlation_r:6.3f} | Skill: {b.skill_score:6.3f}"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
