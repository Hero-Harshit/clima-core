"""
Unit and Integration Test Suite for Phase 4 · Probabilistic Intelligence & Compound Risks.
Validates:
  1. Split Conformal Prediction Intervals & Finite-Sample Marginal Coverage.
  2. Asymmetric & Domain-Specific Conformal Bounds (Temperature, Rain, Power Grid, PM2.5).
  3. Cascading Multi-Hazard Infrastructure Dependency Failure Chains.
  4. Standardized Canonical PredictionResult Contract (v1.0.0) emitted by LiveForecastEngine.
"""

import pytest
import numpy as np
from climacore.analytics.conformal import (
    SplitConformalCalibrator,
    calibrate_temperature_conformal,
    calibrate_rainfall_conformal,
    calibrate_power_demand_conformal,
    calibrate_pm25_conformal,
)
from climacore.models.compound_risk.cascading import CascadingRiskEngine
from climacore.ingestion.engine import LiveForecastEngine
from climacore.ingestion.models import PHYSICAL_UNITS_MANIFEST


def test_split_conformal_calibration_guarantee():
    """Verify Split Conformal Calibrator produces valid intervals with finite-sample coverage."""
    # Synthetic calibration residuals (e.g. from 200 holdout observations)
    rng = np.random.default_rng(42)
    cal_residuals = np.abs(rng.normal(0, 1.5, size=200))

    calibrator = SplitConformalCalibrator(cal_residuals)
    res = calibrator.predict_interval(point_prediction=35.0, alpha=0.10)

    assert res["point_estimate"] == 35.0
    assert res["confidence_level_pct"] == 90.0
    assert res["lower_bound"] < 35.0 < res["upper_bound"]
    assert res["interval_width"] > 0.0
    assert "finite-sample coverage" in res["coverage_guarantee"]

    # Test asymmetric mode
    asym_cal = SplitConformalCalibrator(rng.normal(0.5, 1.2, size=200), asymmetric=True)
    asym_res = asym_cal.predict_interval(point_prediction=20.0, alpha=0.10)
    assert asym_res["lower_bound"] < 20.0 < asym_res["upper_bound"]


def test_domain_conformal_calibrators():
    """Verify domain-specific conformal bounds for temperature, rain, power grid, and PM2.5."""
    # Temperature (44°C daytime peak)
    t_res = calibrate_temperature_conformal(44.0, alpha=0.10)
    assert t_res["lower_bound"] < 44.0 < t_res["upper_bound"]
    assert t_res["interval_width"] <= 8.0

    # Rainfall (bounded below by 0)
    r_res = calibrate_rainfall_conformal(45.0, alpha=0.10)
    assert r_res["lower_bound"] >= 0.0
    assert r_res["upper_bound"] > 45.0

    # Power Grid (MW order)
    p_res = calibrate_power_demand_conformal(12500.0, alpha=0.10)
    assert p_res["lower_bound"] < 12500.0 < p_res["upper_bound"]
    assert p_res["lower_bound"] >= 500.0

    # PM2.5 (bounded below by 1.0)
    aq_res = calibrate_pm25_conformal(180.0, alpha=0.10)
    assert aq_res["lower_bound"] < 180.0 < aq_res["upper_bound"]
    assert aq_res["lower_bound"] >= 1.0


def test_cascading_multi_hazard_chains():
    """Verify CascadingRiskEngine detects multi-hazard causal dependency failures."""
    engine = CascadingRiskEngine()

    # Scenario: Severe Heatwave + Tropical Night in Delhi
    res_heat = engine.evaluate_cascading_chains(
        city="Delhi",
        temp_max_c=45.0,
        temp_min_c=31.0,
        rainfall_mm=0.0,
        pm2_5_ug_m3=80.0,
        is_tropical_night=True,
        grid_deficit_prob=0.45,
        flood_hazard_prob=0.02,
        stagnation_flag=0,
        population_15km=18_000_000,
        dense_cluster_pop_15km=2_800_000,
    )

    assert res_heat["active_cascading_chains_count"] >= 1
    assert any(c["chain_id"] == "CASCADE_THERMAL_GRID_HEALTH" for c in res_heat["active_chains"])
    assert res_heat["cascading_risk_score"] >= 45.0
    assert any("TRANSFORMER" in node["node_id"] for node in res_heat["critical_node_failures"])
    assert len(res_heat["cross_sectoral_interventions"]) > 0

    # Scenario: Cloudburst Deluge in Mumbai
    res_flood = engine.evaluate_cascading_chains(
        city="Mumbai",
        temp_max_c=30.0,
        temp_min_c=25.0,
        rainfall_mm=95.0,
        pm2_5_ug_m3=35.0,
        is_tropical_night=False,
        grid_deficit_prob=0.05,
        flood_hazard_prob=0.75,
        stagnation_flag=0,
        bridges_15km=65,
        drainage_canals_15km=12,
        arterial_roads_15km=140,
    )

    assert res_flood["active_cascading_chains_count"] >= 1
    assert any(c["chain_id"] == "CASCADE_DELUGE_DRAINAGE_TRANSIT" for c in res_flood["active_chains"])
    assert any("UNDERPASS" in node["node_id"] for node in res_flood["critical_node_failures"])
    assert any("Traffic Police" in act or "underpasses" in act for act in res_flood["cross_sectoral_interventions"])


def test_standardized_prediction_result_contract():
    """Verify LiveForecastEngine produces the canonical, complete v1.0.0 PredictionResult contract."""
    engine = LiveForecastEngine()
    result = engine.predict(query="Delhi", persona="all", days=3)

    # 1. Root Contract Metadata
    assert result.schema_version == "v1.0.0"
    assert result.city == "Delhi"
    assert len(result.daily_predictions) == 3

    # 2. System Provenance & Metadata
    assert result.system_provenance is not None
    assert result.system_provenance["engine_version"] == "v1.0.0"
    assert "flood_risk" in result.system_provenance["champion_models"]
    assert "electricity_grid" in result.system_provenance["champion_models"]

    # 3. Spatial Resolution Metadata
    assert result.spatial_resolution_metadata is not None
    assert len(result.spatial_resolution_metadata["concentric_buffers"]) == 3
    assert "WorldPop" in result.spatial_resolution_metadata["demographic_resolution"]

    # 4. Physical Units Manifest
    assert result.physical_units_manifest is not None
    assert result.physical_units_manifest["temp_max"] == "deg_C"
    assert result.physical_units_manifest["precipitation_mm"] == "mm"
    assert result.physical_units_manifest["peak_load_mw"] == "MW"
    assert result.physical_units_manifest["pm2_5"] == "ug/m3"

    # 5. Day Predictions Contract Completeness
    day0 = result.daily_predictions[0]
    assert day0.conformal_bounds is not None
    assert "temperature_c" in day0.conformal_bounds
    assert "rainfall_mm" in day0.conformal_bounds
    assert "peak_grid_load_mw" in day0.conformal_bounds
    assert "pm2_5_ug_m3" in day0.conformal_bounds

    assert day0.cascading_risk is not None
    assert "cascading_risk_score" in day0.cascading_risk
    assert "cascading_severity_tier" in day0.cascading_risk
    assert "cross_sectoral_interventions" in day0.cascading_risk
