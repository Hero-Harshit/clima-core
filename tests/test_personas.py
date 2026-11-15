"""
Unit tests for Clima-Core Persona Intelligence Adapter Engine.
Tests all 6 tailored persona evaluation modules under varying meteorological extremes.
"""

import pytest
from climacore.personas.models import PersonaInput
from climacore.personas.engine import evaluate_persona, evaluate_all_personas


@pytest.fixture
def sample_heatwave_smog_input() -> PersonaInput:
    """Simulate an extreme summer heatwave + smog day in Delhi."""
    return PersonaInput(
        city="Delhi",
        date="2026-05-27",
        temp_max=44.5,
        temp_min=29.0,
        apparent_temp_max=46.5,
        precipitation_mm=0.0,
        rain_probability=0.05,
        pm2_5=220.0,
        pm10=350.0,
        no2=45.0,
        wind_speed_kmh=8.0,
        relative_humidity=45.0,
        solar_radiation=24.0,
        et0_evapotranspiration=7.2,
        imd_heatwave_alert=2,
        atmospheric_stagnation=1,
        historical_percentile=98.5,
    )


@pytest.fixture
def sample_monsoon_downpour_input() -> PersonaInput:
    """Simulate a torrential monsoon downpour in Mumbai."""
    return PersonaInput(
        city="Mumbai",
        date="2026-07-15",
        temp_max=31.0,
        temp_min=25.0,
        apparent_temp_max=36.0,
        precipitation_mm=75.0,
        rain_probability=0.95,
        pm2_5=18.0,
        wind_speed_kmh=28.0,
        relative_humidity=92.0,
        solar_radiation=10.0,
        et0_evapotranspiration=2.8,
        imd_heatwave_alert=0,
        atmospheric_stagnation=0,
        historical_percentile=99.2,
    )


def test_student_persona_heatwave_smog(sample_heatwave_smog_input):
    res = evaluate_persona("student", sample_heatwave_smog_input)
    assert res.persona_id == "student_parent"
    assert res.status_level in ["HIGH_ALERT", "CRITICAL"]
    assert res.impact_score > 60.0
    assert res.key_metrics["morning_assembly_status"] == "MANDATORY_INDOORS"
    assert res.key_metrics["outdoor_sports_allowed"] is False
    assert len(res.operational_restrictions) > 0


def test_student_persona_monsoon(sample_monsoon_downpour_input):
    res = evaluate_persona("student", sample_monsoon_downpour_input)
    assert res.key_metrics["school_bus_flood_risk"] == "HIGH"


def test_farmer_persona(sample_heatwave_smog_input, sample_monsoon_downpour_input):
    # Drought / moisture deficit test
    res_dry = evaluate_persona("farmer", sample_heatwave_smog_input)
    assert res_dry.key_metrics["irrigation_requirement"] == "URGENT"
    assert res_dry.key_metrics["terminal_heat_shock"] is True

    # Rain surplus test
    res_wet = evaluate_persona("farmer", sample_monsoon_downpour_input)
    assert "UNFAVORABLE" in res_wet.key_metrics["spraying_suitability"]
    assert res_wet.key_metrics["irrigation_requirement"] == "NONE_RAIN_SUFFICIENT"


def test_gig_worker_persona_wbgt(sample_heatwave_smog_input):
    res = evaluate_persona("gig_worker", sample_heatwave_smog_input)
    assert res.status_level == "CRITICAL"
    assert res.key_metrics["lost_productive_work_hours"] != "0.0 hours"
    assert "rest" in res.key_metrics["hourly_work_schedule"]


def test_commuter_persona(sample_heatwave_smog_input, sample_monsoon_downpour_input):
    res_vis = evaluate_persona("commuter", sample_heatwave_smog_input)
    assert "CRITICALLY" in res_vis.key_metrics["commute_visibility"] or "REDUCED" in res_vis.key_metrics["commute_visibility"]

    res_flood = evaluate_persona("commuter", sample_monsoon_downpour_input)
    assert "GRIDLOCK" in res_flood.key_metrics["estimated_transit_delay"]


def test_healthcare_persona(sample_heatwave_smog_input):
    res = evaluate_persona("healthcare", sample_heatwave_smog_input)
    assert res.status_level in ["HIGH_ALERT", "CRITICAL"]
    assert res.key_metrics["hospital_surge_alert"] == "HIGH"
    assert "NOT_RECOMMENDED" in res.key_metrics["safe_outdoor_walk_window"]


def test_city_official_persona(sample_heatwave_smog_input):
    res = evaluate_persona("city_official", sample_heatwave_smog_input)
    assert "CRITICAL" in res.key_metrics["power_grid_cooling_stress"] or "ELEVATED" in res.key_metrics["power_grid_cooling_stress"]
    assert res.key_metrics["municipal_action_tier"] == "STAGE_3_FULL_EMERGENCY"


def test_evaluate_all_personas(sample_heatwave_smog_input):
    all_res = evaluate_all_personas(sample_heatwave_smog_input)
    assert len(all_res) == 6
    expected_keys = {
        "student_parent",
        "farmer_kisan",
        "gig_worker_labor",
        "daily_commuter",
        "healthcare_vulnerable",
        "city_official_infrastructure",
    }
    assert set(all_res.keys()) == expected_keys
