"""
Unit & Integration Tests for HydroRIVERS River Hydrology & CGWB Water Quality Intelligence.
Validates:
  1. HydroRIVERS 50-city river network profiles (river distance, Strahler order, discharge, fluvial flood susceptibility).
  2. FloodRisk V3 Fluvial Riverine Flooding interaction & backwater amplification.
  3. CGWB Groundwater Quality feature cleaning, WQI calculation, and BIS IS 10500:2012 exceedances.
  4. WaterQualityRiskEngine ML predictions (hazard probability & continuous WQI).
  5. FastAPI endpoints: /api/predict/water_quality and updated /api/predict/city /custom contracts.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from climacore.features.hydrorivers import get_city_river_profile
from climacore.features.groundwater_quality import (
    compute_water_quality_index,
    BIS_STANDARDS,
)
from climacore.models.flood_risk.engine import FloodRiskEngine
from climacore.models.water_quality.engine import WaterQualityRiskEngine
from climacore.service.app import app


@pytest.fixture
def api_client() -> TestClient:
    return TestClient(app)


# ----------------------------------------------------------------------
# 1. HydroRIVERS Hydrology Profile Tests
# ----------------------------------------------------------------------
def test_hydrorivers_city_profile_extraction():
    """Verify HydroRIVERS profiles for key Indian riverine cities."""
    delhi_prof = get_city_river_profile("Delhi")
    assert delhi_prof["city"] == "Delhi"
    assert "nearest_river" in delhi_prof
    assert "nearest_major_river" in delhi_prof
    assert delhi_prof["nearest_river"]["distance_km"] < 10.0  # Yamuna river proximity
    assert delhi_prof["nearest_major_river"]["average_discharge_m3s"] > 50.0  # Major Yamuna discharge
    assert delhi_prof["fluvial_flood_susceptibility_tier"] in ["HIGH", "VERY_HIGH"]

    kolkata_prof = get_city_river_profile("Kolkata")
    assert kolkata_prof["nearest_major_river"]["average_discharge_m3s"] > 200.0  # Massive Hooghly flow
    assert kolkata_prof["fluvial_flood_susceptibility_tier"] == "VERY_HIGH"


def test_flood_risk_engine_fluvial_integration():
    """Verify that FloodRiskEngine incorporates HydroRIVERS fluvial backwater amplification."""
    engine = FloodRiskEngine()

    # Heavy 3-day rainfall in riverine city (Kolkata)
    res_riverine = engine.evaluate_flood_risk(
        city="Kolkata",
        rainfall_mm=85.0,
        rain_rolling_3d=140.0,
        smdi=1.5,
        forecast_horizon_hours=48,
    )

    assert "river_hydrology" in res_riverine
    hyd = res_riverine["river_hydrology"]
    assert hyd["nearest_major_river_distance_km"] < 5.0
    assert hyd["fluvial_flood_susceptibility_tier"] == "VERY_HIGH"
    assert hyd["fluvial_amplification_factor"] > 1.05  # Fluvial backwater amplifies 48h hazard
    assert res_riverine["multi_horizon_hazard"]["p_48h_channel_overflow"] > 0.40


# ----------------------------------------------------------------------
# 2. CGWB Groundwater Quality & BIS Exceedance Tests
# ----------------------------------------------------------------------
def test_wqi_calculation_safe_water():
    """Verify WQI calculation for clean, within-limits potable groundwater."""
    sample_safe = {
        "ph": 7.2,
        "total_dissolved_solids": 280.0,
        "total_hardness": 140.0,
        "chloride": 60.0,
        "sulphate": 45.0,
        "calcium": 40.0,
        "magnesium": 18.0,
    }
    wqi, status = compute_water_quality_index(sample_safe)
    assert wqi < 50.0
    assert status in ["EXCELLENT", "GOOD"]


def test_wqi_calculation_contaminated_water():
    """Verify WQI calculation for heavily contaminated, non-potable groundwater."""
    sample_toxic = {
        "ph": 9.2,  # Abnormal alkaline pH
        "total_dissolved_solids": 2400.0,  # Exceeds permissible 2000 mg/L
        "total_hardness": 850.0,  # Exceeds permissible 600 mg/L
        "chloride": 650.0,  # High salinity
        "sulphate": 450.0,  # Exceeds permissible 400 mg/L
        "calcium": 180.0,
        "magnesium": 95.0,
    }
    wqi, status = compute_water_quality_index(sample_toxic)
    assert wqi > 100.0
    assert status == "UNSUITABLE_FOR_DRINKING"


# ----------------------------------------------------------------------
# 3. WaterQualityRiskEngine Model Evaluation Tests
# ----------------------------------------------------------------------
def test_water_quality_risk_engine():
    """Verify WaterQualityRiskEngine returns calibrated hazard probability and health advisories."""
    engine = WaterQualityRiskEngine()

    res = engine.evaluate_water_quality(
        latitude=19.0760,
        longitude=72.8777,
        city="Mumbai",
        ph=8.8,
        electric_conductivity=3200.0,  # High EC (~2048 mg/L TDS)
        total_hardness=720.0,
    )

    assert res["city"] == "Mumbai"
    assert "contamination_hazard_probability" in res
    assert res["contamination_hazard_probability"] > 0.50
    assert res["wqi_grade"] in ["VERY_POOR", "UNSUITABLE_FOR_DRINKING"]
    assert len(res["bis_standard_exceedances"]) > 0
    assert len(res["operational_advisories"]) > 0
    assert any("RO" in adv or "Reverse Osmosis" in adv for adv in res["operational_advisories"])


# ----------------------------------------------------------------------
# 4. FastAPI Endpoint Integration Tests
# ----------------------------------------------------------------------
def test_api_predict_water_quality_endpoint(api_client: TestClient):
    """Test dedicated /api/predict/water_quality endpoint."""
    payload = {
        "latitude": 18.5204,
        "longitude": 73.8567,
        "city": "Pune",
        "ph": 7.6,
        "electric_conductivity": 850.0,
        "total_hardness": 280.0,
        "is_drought_stressed": False,
    }
    r = api_client.post("/api/predict/water_quality", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["city"] == "Pune"
    assert "contamination_hazard_probability" in data
    assert "wqi_grade" in data
    assert "estimated_parameters" in data

    # Warm execution should be sub-50ms
    r_warm = api_client.post("/api/predict/water_quality", json=payload)
    assert r_warm.status_code == 200
    assert r_warm.json()["latency_ms"] < 50.0


def test_api_predict_city_includes_water_quality_and_river(api_client: TestClient):
    """Test that live city forecasts now include water_quality_risk and river hydrology."""
    r = api_client.post("/api/predict/city", json={"city": "Delhi"})
    assert r.status_code == 200
    data = r.json()

    first_day = data["daily_predictions"][0]
    assert "water_quality_risk" in first_day
    wq = first_day["water_quality_risk"]
    assert "contamination_hazard_probability" in wq
    assert "wqi_grade" in wq

    # Check flood risk river hydrology
    flood = first_day["flagship_intelligence"]["flood_risk"]
    assert "river_hydrology" in flood
    assert flood["river_hydrology"]["nearest_river_distance_km"] < 10.0
