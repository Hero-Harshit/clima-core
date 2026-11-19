"""Unit tests for TreeSHAP Explanation Engine V1."""

from pathlib import Path
import lightgbm as lgb
import numpy as np
import pytest

from climacore.explainability.treeshap import (
    OutputSpace,
    TreeSHAPEngine,
    TreeSHAPExplanation,
)


def test_treeshap_additivity_flood_risk_champion():
    model_path = Path("artifacts/models/flood_risk/v0.1.0/flood_risk_v1.booster")
    assert model_path.exists()

    engine = TreeSHAPEngine()
    booster = engine.get_booster(model_path)
    feature_names = booster.feature_name()

    # Synthetic realistic scenario
    features = {
        "rainfall_mm": 125.0,
        "rain_rolling_3d": 210.0,
        "rain_rolling_7d": 350.0,
        "rain_rolling_14d": 400.0,
        "rain_rolling_30d": 600.0,
        "rain_max_3d": 125.0,
        "api_14d": 180.0,
        "precipitation_hours": 18.0,
        "et0_fao_evapotranspiration": 2.5,
        "wind_gusts_10m_max": 65.0,
        "soil_moisture_0_to_7cm_mean": 0.45,
        "soil_moisture_7_to_28cm_mean": 0.42,
        "topsoil_moisture_anomaly": 2.8,
        "rootzone_moisture_anomaly": 2.5,
        "soil_moisture_deficit_index": -2.0,
        "drainage_canals_15km": 12.0,
        "bridges_15km": 8.0,
        "transit_stations_15km": 15.0,
        "population_15km": 1200000.0,
        "dense_cluster_pop_15km": 850000.0,
    }

    explanation = engine.explain(
        model=booster,
        features=features,
        model_id="flood_risk_v1",
        model_version="v0.1.0",
        is_classifier=True,
        target_unit="log-odds",
    )

    assert isinstance(explanation, TreeSHAPExplanation)
    assert explanation.output_space == OutputSpace.LOG_ODDS_MARGIN
    assert explanation.target_unit == "log-odds"

    # Mathematical additivity check: error must be under 1e-6
    assert explanation.additivity_error < 1e-6

    # Probability check: transformed_prediction must be between 0 and 1
    assert explanation.transformed_prediction is not None
    assert 0.0 <= explanation.transformed_prediction <= 1.0

    # Top drivers check
    assert len(explanation.top_positive_contributions) > 0
    top_pos = explanation.top_positive_contributions[0]
    assert top_pos.shap_value > 0.0
    assert top_pos.direction == "amplifies"
    assert top_pos.unit != ""

    # Latency benchmark: must be well under 2ms
    assert explanation.latency_ms < 50.0  # Cold/warm threshold
    assert "DO NOT represent causal counterfactual effects" in explanation.non_causal_disclaimer


def test_treeshap_regression_model_electricity_demand():
    model_path = Path("artifacts/models/electricity_grid/v0.1.0/peak_grid_load_v1.booster")
    assert model_path.exists()

    engine = TreeSHAPEngine()
    booster = engine.get_booster(model_path)
    feature_names = booster.feature_name()

    sample_features = {f: 1.0 for f in feature_names}
    explanation = engine.explain(
        model=booster,
        features=sample_features,
        model_id="peak_grid_load_v1",
        model_version="v0.1.0",
        is_classifier=False,
        target_unit="MW",
    )

    assert explanation.output_space == OutputSpace.PHYSICAL_UNITS
    assert explanation.target_unit == "MW"
    assert explanation.transformed_prediction is None
    assert explanation.additivity_error < 1e-6


def test_treeshap_latency_benchmark():
    model_path = Path("artifacts/models/flood_risk/v0.1.0/flood_risk_v1.booster")
    engine = TreeSHAPEngine()
    booster = engine.get_booster(model_path)
    feature_names = booster.feature_name()
    features = {f: 10.0 for f in feature_names}

    # Warmup
    _ = engine.explain(booster, features, is_classifier=True)

    # 10 iterations
    latencies = []
    for _ in range(10):
        exp = engine.explain(booster, features, is_classifier=True)
        latencies.append(exp.latency_ms)

    avg_latency = sum(latencies) / len(latencies)
    assert avg_latency < 2.0  # Must be sub-2ms
