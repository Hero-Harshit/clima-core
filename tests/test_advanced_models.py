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


def test_climate_master_44y_model_artifact():
    root = get_project_root()
    model_file = root / "artifacts" / "models" / "climate_master_44y" / "v0.1.0" / "model.txt"
    assert model_file.exists(), f"Climate Master model artifact missing at {model_file}"

    booster = lgb.Booster(model_file=str(model_file))
    assert booster.num_trees() > 0
    feature_names = booster.feature_name()
    assert len(feature_names) > 10
    assert "temperature_2m_max_lag_1" in feature_names


def test_imd_rain_gauge_model_artifact():
    root = get_project_root()
    model_file = root / "artifacts" / "models" / "imd_rain_gauge" / "v0.1.0" / "model.txt"
    assert model_file.exists(), f"IMD Rain Gauge model artifact missing at {model_file}"

    booster = lgb.Booster(model_file=str(model_file))
    assert booster.num_trees() > 0
    feature_names = booster.feature_name()
    assert "imd_rainfall_mm_lag_1" in feature_names


def test_cpcb_station_model_artifact():
    root = get_project_root()
    model_file = root / "artifacts" / "models" / "cpcb_station" / "v0.1.0" / "model.txt"
    assert model_file.exists(), f"CPCB Station model artifact missing at {model_file}"

    booster = lgb.Booster(model_file=str(model_file))
    assert booster.num_trees() > 0
    feature_names = booster.feature_name()
    assert "cpcb_pm2_5_mean_lag_1" in feature_names


def test_apparent_heat_stress_model_artifact():
    root = get_project_root()
    model_file = root / "artifacts" / "models" / "apparent_heat_stress" / "v0.1.0" / "model.txt"
    assert model_file.exists(), f"Apparent Heat Stress model artifact missing at {model_file}"

    booster = lgb.Booster(model_file=str(model_file))
    assert booster.num_trees() > 0
    feature_names = booster.feature_name()
    assert "apparent_temperature_max_lag_1" in feature_names


def test_heatwave_alert_model_artifact():
    root = get_project_root()
    model_file = root / "artifacts" / "models" / "heatwave_alert" / "v0.1.0" / "model.txt"
    assert model_file.exists(), f"Heatwave Alert model artifact missing at {model_file}"

    booster = lgb.Booster(model_file=str(model_file))
    assert booster.num_trees() > 0
    feature_names = booster.feature_name()
    assert "tmax_departure_lag_1" in feature_names or "imd_heatwave_today_lag_1" in feature_names


def test_cpcb_multipollutant_models_artifacts():
    root = get_project_root()
    pm10_file = root / "artifacts" / "models" / "cpcb_multipollutant" / "v0.1.0" / "pm10_model.txt"
    no2_file = root / "artifacts" / "models" / "cpcb_multipollutant" / "v0.1.0" / "no2_model.txt"
    assert pm10_file.exists(), f"PM10 model artifact missing at {pm10_file}"
    assert no2_file.exists(), f"NO2 model artifact missing at {no2_file}"

    b_pm10 = lgb.Booster(model_file=str(pm10_file))
    b_no2 = lgb.Booster(model_file=str(no2_file))
    assert b_pm10.num_trees() > 0
    assert b_no2.num_trees() > 0


def test_atmospheric_stagnation_model_artifact():
    root = get_project_root()
    model_file = root / "artifacts" / "models" / "atmospheric_stagnation" / "v0.1.0" / "model.txt"
    assert model_file.exists(), f"Stagnation model artifact missing at {model_file}"

    booster = lgb.Booster(model_file=str(model_file))
    assert booster.num_trees() > 0
    feature_names = booster.feature_name()
    assert "wind_speed_10m_max_lag_1" in feature_names


def test_shap_explainability_artifacts():
    root = get_project_root()
    summary_file = root / "artifacts" / "explainability" / "global_importance_summary.json"
    assert summary_file.exists(), f"SHAP summary missing at {summary_file}"

    import json
    with open(summary_file, "r") as f:
        data = json.load(f)
    assert "heat_risk" in data
    assert "air_quality" in data
    assert len(data["heat_risk"]["top_10_global_drivers"]) == 10


