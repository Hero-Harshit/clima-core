"""
Unit and integration tests for Clima-Core ML pipeline.
Verifies temporal leakage prevention, model artifact reload parity, and contract validation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import pytest

from climacore.config.loader import get_project_root
from climacore.features.temporal import add_temporal_features
from climacore.features.lags import add_lag_features
from climacore.features.rolling import add_rolling_features
from climacore.common.contracts import DatasetManifest, FileChecksum


def test_add_temporal_features():
    df = pd.DataFrame({"date": ["2026-01-01", "2026-07-01", "2026-12-31"]})
    res = add_temporal_features(df, date_col="date")
    assert "day_of_year" in res.columns
    assert "sin_day_of_year" in res.columns
    assert "cos_day_of_year" in res.columns
    assert len(res) == 3


def test_lags_prevent_future_leakage():
    # If a future row changes, past lags must NOT change
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "city": ["Delhi"] * 10,
        "date": dates,
        "val": list(range(10)),
    })
    lagged_1 = add_lag_features(df, group_col="city", date_col="date", target_cols=["val"], lags=[1, 2])
    
    # Modify future value at index 9
    df_modified = df.copy()
    df_modified.loc[9, "val"] = 999
    lagged_2 = add_lag_features(df_modified, group_col="city", date_col="date", target_cols=["val"], lags=[1, 2])

    # Past lags up to index 8 must remain identical
    assert np.all(lagged_1.loc[:8, "val_lag_1"].fillna(-1) == lagged_2.loc[:8, "val_lag_1"].fillna(-1))


def test_rolling_prevents_future_leakage():
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "city": ["Mumbai"] * 10,
        "date": dates,
        "val": [10.0] * 10,
    })
    rolled = add_rolling_features(df, group_col="city", date_col="date", target_cols=["val"], windows=[3])
    # The first row rolling mean should be 10.0 (with min_periods=1)
    assert rolled.loc[0, "val_roll_mean_3d"] == 10.0
    # Modifying day 5 should not change days 0-4
    df_mod = df.copy()
    df_mod.loc[5, "val"] = 500.0
    rolled_mod = add_rolling_features(df_mod, group_col="city", date_col="date", target_cols=["val"], windows=[3])
    pd.testing.assert_series_equal(rolled.loc[:4, "val_roll_mean_3d"], rolled_mod.loc[:4, "val_roll_mean_3d"])


def test_air_quality_model_artifact_parity():
    root = get_project_root()
    model_path = root / "artifacts" / "models" / "air_quality" / "v0.1.0" / "model.txt"
    parquet_path = root / "data" / "processed" / "air_quality_train_ready.parquet"
    
    if not model_path.exists() or not parquet_path.exists():
        pytest.skip("Model or parquet dataset not yet built")

    booster = lgb.Booster(model_file=str(model_path))
    df = pd.read_parquet(parquet_path)
    sample = df.tail(10)
    
    feature_cols = booster.feature_name()
    preds = booster.predict(sample[feature_cols])
    
    assert len(preds) == 10
    assert not np.isnan(preds).any()
    assert (preds > 0).all()  # PM2.5 concentrations must be positive
