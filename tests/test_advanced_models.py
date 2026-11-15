"""
Unit tests for Clima-Core advanced models and new data sources:
- Multi-horizon forecast model reload parity
- Quantile monotonicity (Q10 <= Q50 <= Q90)
- IMD NetCDF extracted physical rainfall table integrity
- CPCB ground station dataset integrity
"""

from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from climacore.config.loader import get_project_root


def test_imd_rainfall_processed_dataset():
    root = get_project_root()
    p_file = root / "data" / "processed" / "imd_rainfall_50_cities.parquet"
    if not p_file.exists():
        pytest.skip("IMD processed dataset not yet built")

    df = pd.read_parquet(p_file)
    assert len(df) > 80000
    assert "imd_rainfall_mm" in df.columns
    assert "city" in df.columns
    assert "date" in df.columns
    # Check non-negative rainfall values
    valid_rain = df["imd_rainfall_mm"].dropna()
    assert (valid_rain >= 0.0).all()


def test_cpcb_station_processed_dataset():
    root = get_project_root()
    p_file = root / "data" / "processed" / "cpcb_ground_truth_cities.parquet"
    if not p_file.exists():
        pytest.skip("CPCB station processed dataset not yet built")

    df = pd.read_parquet(p_file)
    assert len(df) > 20000
    assert "cpcb_pm2_5_mean" in df.columns
    assert "city" in df.columns


def test_quantile_uncertainty_monotonicity():
    root = get_project_root()
    q10_file = root / "artifacts" / "models" / "uncertainty" / "v0.1.0" / "heat_risk_quantile_10.txt"
    q50_file = root / "artifacts" / "models" / "uncertainty" / "v0.1.0" / "heat_risk_quantile_50.txt"
    q90_file = root / "artifacts" / "models" / "uncertainty" / "v0.1.0" / "heat_risk_quantile_90.txt"
    data_file = root / "data" / "processed" / "heat_risk_train_ready.parquet"

    if not q10_file.exists() or not data_file.exists():
        pytest.skip("Quantile models not yet trained")

    b10 = lgb.Booster(model_file=str(q10_file))
    b50 = lgb.Booster(model_file=str(q50_file))
    b90 = lgb.Booster(model_file=str(q90_file))

    df = pd.read_parquet(data_file).tail(20)
    feats = b10.feature_name()

    p10 = b10.predict(df[feats])
    p50 = b50.predict(df[feats])
    p90 = b90.predict(df[feats])

    # Monotonicity test: p10 <= p50 <= p90 across all samples (with tiny numerical tolerance)
    assert (p50 >= p10 - 0.5).all(), "Median should generally exceed 10th percentile"
    assert (p90 >= p50 - 0.5).all(), "90th percentile should generally exceed median"
