"""
Unit tests for Clima-Core Persona Intelligence Adapter Engine.
Tests all 13 tailored persona evaluation modules under varying meteorological extremes:
Student, Farmer, Gig Worker, Commuter, Healthcare, City Official,
Maritime, Industry, Aviation, Water, Insurance, Cold Chain, and Tourism.
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
        wet_bulb_temp_c=31.2,
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
        soil_moisture_deficit_index=1.6,
    )


@pytest.fixture
def sample_cyclone_gale_input() -> PersonaInput:
    """Simulate a severe cyclonic storm approaching Chennai."""
    return PersonaInput(
        city="Chennai",
        date="2026-11-20",
        temp_max=28.5,
        temp_min=24.0,
        precipitation_mm=120.0,
        rain_probability=0.98,
        pm2_5=12.0,
        wind_speed_kmh=68.0,
        gale_wind_gust_kmh=95.0,
        cyclone_distance_km=75.0,
        cyclone_grade="VSCS",
        relative_humidity=95.0,
        historical_percentile=99.8,
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
    res_dry = evaluate_persona("farmer", sample_heatwave_smog_input)
    assert res_dry.key_metrics["irrigation_requirement"] == "URGENT"
    assert res_dry.key_metrics["terminal_heat_shock"] is True

    res_wet = evaluate_persona("farmer", sample_monsoon_downpour_input)
    assert "UNFAVORABLE" in res_wet.key_metrics["spraying_suitability"]
    assert res_wet.key_metrics["irrigation_requirement"] == "NONE_RAIN_SUFFICIENT"


def test_farmer_flash_drought_alert():
    inp = PersonaInput(
        city="Nagpur",
        date="2026-05-15",
        temp_max=38.0,
        pm2_5=45.0,
        soil_moisture_deficit_index=-2.1,
        flash_drought_alert=1,
    )
    res = evaluate_persona("farmer", inp)
    assert res.status_level == "CRITICAL"
    assert res.key_metrics["flash_drought_alert"] == "ACTIVE_CRITICAL"
    assert res.key_metrics["irrigation_requirement"] == "EMERGENCY"
    assert any("FLASH DROUGHT" in r for r in res.operational_restrictions)


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


def test_maritime_persona(sample_cyclone_gale_input):
    res = evaluate_persona("maritime", sample_cyclone_gale_input)
    assert res.persona_id == "maritime_coastal"
    assert res.status_level == "CRITICAL"
    assert res.key_metrics["container_crane_operations"] == "SHUTDOWN_MANDATED"
    assert res.key_metrics["coastal_fishing_clearance"] == "SAILING_PROHIBITED"
    assert "DIRECT_STRIKE_ZONE" in res.key_metrics["cyclone_proximity_status"]
    assert any("prohibition" in r.lower() for r in res.operational_restrictions)


def test_industry_persona(sample_heatwave_smog_input):
    res = evaluate_persona("industry", sample_heatwave_smog_input)
    assert res.persona_id == "industry_manufacturing"
    assert res.status_level in ["HIGH_ALERT", "CRITICAL"]
    assert "STACK_TRAPPING" in res.key_metrics["stack_plume_dispersion"] or "POOR" in res.key_metrics["stack_plume_dispersion"]
    assert "CRITICAL" in res.key_metrics["cooling_tower_status"] or "ELEVATED" in res.key_metrics["cooling_tower_status"]
    assert any("derating" in r.lower() or "banned" in r.lower() for r in res.operational_restrictions)


def test_aviation_persona(sample_heatwave_smog_input):
    res = evaluate_persona("aviation", sample_heatwave_smog_input)
    assert res.persona_id == "aviation_airport"
    assert res.key_metrics["density_altitude_status"] == "EXTREME_DENSITY_ALTITUDE_PENALTY"
    assert "HAZARDOUS" in res.key_metrics["apron_tarmac_surface_temp"]
    assert any("MTOW" in r for r in res.operational_restrictions)


def test_water_persona(sample_monsoon_downpour_input):
    res = evaluate_persona("water", sample_monsoon_downpour_input)
    assert res.persona_id == "water_utilities"
    assert res.status_level in ["HIGH_ALERT", "CRITICAL"]
    assert res.key_metrics["soil_saturation_state"].startswith("FULLY_SATURATED")
    assert "CRITICAL" in res.key_metrics["stormwater_drain_status"] or "ELEVATED" in res.key_metrics["stormwater_drain_status"]


def test_insurance_persona(sample_cyclone_gale_input):
    res = evaluate_persona("insurance", sample_cyclone_gale_input)
    assert res.persona_id == "insurance_actuarial"
    assert res.status_level == "CRITICAL"
    assert "CATASTROPHIC_STRIKE" in res.key_metrics["cyclone_catastrophe_status"]
    assert res.key_metrics["claims_processing_mode"] == "AUTOMATED_PARAMETRIC_EXECUTION"


def test_cold_chain_persona(sample_heatwave_smog_input):
    res = evaluate_persona("cold_chain", sample_heatwave_smog_input)
    assert res.persona_id == "cold_chain_logistics"
    assert res.status_level in ["HIGH_ALERT", "CRITICAL"]
    assert "EXTREME" in res.key_metrics["compressor_operating_status"] or "ELEVATED" in res.key_metrics["compressor_operating_status"]
    assert "NOCTURNAL_DISPATCH_MANDATED" in res.key_metrics["recommended_dispatch_window"]


def test_tourism_persona(sample_heatwave_smog_input):
    res = evaluate_persona("tourism", sample_heatwave_smog_input)
    assert res.persona_id == "tourism_hospitality"
    assert res.status_level in ["HIGH_ALERT", "CRITICAL"]
    assert "HAZARDOUS" in res.key_metrics["heritage_monument_surface_temp"] or "ELEVATED" in res.key_metrics["heritage_monument_surface_temp"]
    assert "CRITICAL" in res.key_metrics["pilgrim_queue_heat_status"] or "MODERATE" in res.key_metrics["pilgrim_queue_heat_status"]


def test_evaluate_all_13_personas(sample_heatwave_smog_input):
    all_res = evaluate_all_personas(sample_heatwave_smog_input)
    assert len(all_res) == 13
    expected_keys = {
        "student_parent",
        "farmer_kisan",
        "gig_worker_labor",
        "daily_commuter",
        "healthcare_vulnerable",
        "city_official_infrastructure",
        "maritime_coastal",
        "industry_manufacturing",
        "aviation_airport",
        "water_utilities",
        "insurance_actuarial",
        "cold_chain_logistics",
        "tourism_hospitality",
    }
    assert set(all_res.keys()) == expected_keys
