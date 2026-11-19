"""Unit tests for ForecastCorrectionEngine V1."""

import numpy as np
import pandas as pd
import pytest

from climacore.analytics.forecast_correction import (
    CorrectionMethod,
    ForecastCorrectionEngine,
    ForecastVariable,
    MeanBiasCorrection,
    QuantileMappingCorrection,
)


def test_mean_bias_correction_additive():
    df = pd.DataFrame({
        "t2m_fcst": [30.0, 32.0, 35.0, 28.0],
        "t2m_obs": [29.0, 31.0, 34.0, 27.0],
        "city": ["delhi", "delhi", "delhi", "delhi"],
        "season": ["summer", "summer", "summer", "summer"],
    })
    mbe = MeanBiasCorrection(ForecastVariable.T2M_MAX, additive=True)
    mbe.fit(df, "t2m_fcst", "t2m_obs")
    assert pytest.approx(mbe.global_offset, abs=1e-3) == 1.0

    # Test correction
    corrected = mbe.predict(33.0)
    assert pytest.approx(corrected, abs=1e-3) == 32.0


def test_mean_bias_correction_multiplicative_precipitation():
    df = pd.DataFrame({
        "precip_fcst": [10.0, 20.0, 30.0],
        "precip_obs": [5.0, 10.0, 15.0],
    })
    mbe = MeanBiasCorrection(ForecastVariable.PRECIPITATION, additive=False)
    mbe.fit(df, "precip_fcst", "precip_obs")
    # Ratio is ~ 0.5
    assert pytest.approx(mbe.global_offset, abs=0.05) == 0.5

    corrected = mbe.predict(40.0)
    assert pytest.approx(corrected, abs=0.5) == 20.0
    # Must never be negative
    assert mbe.predict(0.0) >= 0.0


def test_quantile_mapping_correction():
    np.random.seed(42)
    # NWP forecast systematically cold by 2 degrees
    obs = np.random.normal(loc=30.0, scale=3.0, size=500)
    fcst = obs - 2.0

    qm = QuantileMappingCorrection(ForecastVariable.T2M_MAX)
    qm.fit(fcst, obs)

    # Test mapping
    test_val = 28.0  # corresponds to obs 30.0
    mapped = qm.predict(test_val)
    assert pytest.approx(mapped, abs=0.5) == 30.0


def test_quantile_mapping_precipitation_wet_threshold():
    # Dry days must map to 0.0 without drizzle artifact
    fcst = np.array([0.0, 0.05, 5.0, 15.0, 30.0, 0.02])
    obs = np.array([0.0, 0.0, 6.0, 14.0, 32.0, 0.0])

    qm = QuantileMappingCorrection(ForecastVariable.PRECIPITATION, wet_threshold=0.1)
    qm.fit(fcst, obs)

    assert qm.predict(0.0) == 0.0
    assert qm.predict(0.05) == 0.0
    assert qm.predict(15.0) > 0.0


def test_forecast_correction_engine_end_to_end():
    engine = ForecastCorrectionEngine()

    dates = pd.date_range("2024-01-01", periods=100, freq="D").astype(str)
    np.random.seed(42)
    df = pd.DataFrame({
        "date": dates,
        "city": ["delhi"] * 100,
        "season": ["winter"] * 50 + ["summer"] * 50,
        "t2m_min_fcst": np.random.uniform(15, 25, 100),
        "t2m_min_obs": np.random.uniform(14, 24, 100),
        "t2m_max_fcst": np.random.uniform(30, 42, 100),
        "rh_mean_fcst": np.random.uniform(40, 80, 100),
        "ws_mean_fcst": np.random.uniform(5, 20, 100),
        "sin_doy": np.zeros(100),
        "cos_doy": np.ones(100),
        "elevation": [216.0] * 100,
        "latitude": [28.6] * 100,
        "longitude": [77.2] * 100,
    })

    feature_cols = ["t2m_min_fcst", "rh_mean_fcst", "ws_mean_fcst", "elevation"]
    benchmarks = engine.train_and_benchmark(
        df=df,
        variable=ForecastVariable.T2M_MIN,
        fcst_col="t2m_min_fcst",
        obs_col="t2m_min_obs",
        feature_cols=feature_cols,
        train_cutoff_date=dates[70],
    )

    assert len(benchmarks) == 4
    methods = [b.method for b in benchmarks]
    assert CorrectionMethod.RAW in methods
    assert CorrectionMethod.MEAN_BIAS in methods
    assert CorrectionMethod.QUANTILE_MAPPING in methods
    assert CorrectionMethod.ML_RESIDUAL in methods

    # Test calibrate API
    cal = engine.calibrate(
        variable=ForecastVariable.T2M_MIN,
        raw_val=20.0,
        features={"t2m_min_fcst": 20.0, "rh_mean_fcst": 50.0, "ws_mean_fcst": 10.0, "elevation": 216.0},
        method=CorrectionMethod.ML_RESIDUAL,
    )
    assert cal.variable == "t2m_min"
    assert cal.raw_value == 20.0
    assert cal.method == CorrectionMethod.ML_RESIDUAL
    assert cal.lower_ci_90 <= cal.corrected_value <= cal.upper_ci_90
