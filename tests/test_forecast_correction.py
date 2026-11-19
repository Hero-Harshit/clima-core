"""Unit tests for ForecastCorrectionEngine V1 (Scientific Validation & Integrity)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from climacore.analytics.forecast_correction import (
    CalibrationDecision,
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

    corrected = mbe.predict(33.0)
    assert pytest.approx(corrected, abs=1e-3) == 32.0


def test_mean_bias_correction_multiplicative_precipitation():
    df = pd.DataFrame({
        "precip_fcst": [10.0, 20.0, 30.0],
        "precip_obs": [5.0, 10.0, 15.0],
    })
    mbe = MeanBiasCorrection(ForecastVariable.PRECIPITATION, additive=False)
    mbe.fit(df, "precip_fcst", "precip_obs")
    assert pytest.approx(mbe.global_offset, abs=0.05) == 0.5

    corrected = mbe.predict(40.0)
    assert pytest.approx(corrected, abs=0.5) == 20.0
    assert mbe.predict(0.0) >= 0.0


def test_quantile_mapping_correction():
    np.random.seed(42)
    obs = np.random.normal(loc=30.0, scale=3.0, size=500)
    fcst = obs - 2.0

    qm = QuantileMappingCorrection(ForecastVariable.T2M_MAX)
    qm.fit(fcst, obs)

    test_val = 28.0
    mapped = qm.predict(test_val)
    assert pytest.approx(mapped, abs=0.5) == 30.0


def test_quantile_mapping_precipitation_wet_threshold():
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


def test_canonical_verification_schema_integrity():
    canonical_path = Path("data/processed/forecast_correction_pairs/canonical_forecast_verification_pairs.parquet")
    assert canonical_path.exists()
    df_canon = pd.read_parquet(canonical_path)

    required_cols = [
        "location_id", "latitude", "longitude", "forecast_model",
        "forecast_issue_time", "valid_time", "forecast_lead_hours",
        "variable", "forecast_value", "observed_value",
        "observation_source", "observation_quality_flag", "unit"
    ]
    for c in required_cols:
        assert c in df_canon.columns, f"Missing required canonical column: {c}"

    assert len(df_canon) > 1000
    assert set(df_canon["observation_quality_flag"].unique()) == {"QC_PASS_INDEPENDENT"}
    assert "IMD_Gridded_Obs" in df_canon["observation_source"].iloc[0]


def test_precipitation_all_day_vs_conditional_evaluation():
    engine = ForecastCorrectionEngine()
    obs = np.array([0.0, 0.0, 0.0, 5.0, 15.0, 30.0])
    fcst = np.array([0.0, 0.5, 0.0, 4.0, 10.0, 25.0])

    contingency = engine.compute_precipitation_contingency(obs, fcst, threshold_mm=0.1)
    assert contingency.observed_wet_days == 3
    assert contingency.forecast_wet_days == 4
    assert contingency.recall == 1.0  # Caught all 3 wet days
    assert 0.0 <= contingency.threat_score_csi <= 1.0
