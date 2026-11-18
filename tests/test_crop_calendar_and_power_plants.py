"""
Unit and integration tests for SAGE Crop Calendar Phenology and CHETNA Industrial Power Plants.
Validates phenological growth-stage mapping, industrial point-source exposure metrics,
and persona alert activations.
"""

import json
from pathlib import Path
import pytest
from climacore.config.loader import get_project_root
from climacore.personas.models import PersonaInput
from climacore.personas.farmer import evaluate_farmer_persona
from climacore.personas.city_official import evaluate_city_official_persona
from climacore.personas.industry import evaluate_industry_persona
from climacore.ingestion.engine import LiveForecastEngine


@pytest.fixture(scope="module")
def project_root() -> Path:
    return get_project_root()


def test_crop_calendar_profiles_integrity(project_root):
    """Verify city_crop_calendars.json covers 50 cities with valid DOY ranges."""
    json_path = project_root / "data" / "processed" / "agriculture" / "city_crop_calendars.json"
    assert json_path.exists(), f"Missing crop calendar json at {json_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    assert len(profiles) >= 50, f"Expected 50 city profiles, found {len(profiles)}"
    for p in profiles:
        assert "city" in p
        assert "kharif_crop" in p
        assert "kharif_sowing_doy_range" in p
        assert len(p["kharif_sowing_doy_range"]) == 2
        assert "kharif_anthesis_doy_range" in p
        assert "kharif_harvest_doy_range" in p
        assert "rabi_crop" in p


def test_chetna_power_plants_exposure_integrity(project_root):
    """Verify city_power_plant_exposure.json covers 50 cities with valid capacity and distance."""
    json_path = project_root / "data" / "processed" / "power_plants" / "city_power_plant_exposure.json"
    assert json_path.exists(), f"Missing power plant exposure json at {json_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    assert len(profiles) >= 50, f"Expected 50 city profiles, found {len(profiles)}"
    delhi_prof = next((p for p in profiles if p["city"].lower() == "delhi"), None)
    assert delhi_prof is not None
    assert delhi_prof["thermal_capacity_100km_mw"] >= 15000.0, f"Delhi thermal 100km MW too low: {delhi_prof}"
    assert delhi_prof["closest_plant_distance_km"] <= 10.0


def test_farmer_phenological_anthesis_heat_alert():
    """Verify Farmer persona triggers PHENOLOGICAL ANTHESIS ALERT during flowering when temp >= 32C."""
    inp = PersonaInput(
        city="Ludhiana",
        date="2026-02-15",
        temp_max=34.0,
        temp_min=14.0,
        apparent_temp_max=33.0,
        precipitation_mm=0.0,
        rain_probability=0.0,
        pm2_5=85.0,
        wind_speed_kmh=8.0,
        relative_humidity=50.0,
        solar_radiation=18.0,
        et0_evapotranspiration=4.2,
        crop_growth_stage="Rabi Wheat: Anthesis / Flowering",
        crop_anthesis_dates="Feb 10 - Mar 05",
    )

    result = evaluate_farmer_persona(inp)
    advisories_str = " ".join(result.advisories)
    assert "PHENOLOGICAL ANTHESIS ALERT" in advisories_str
    assert "floret abortion risk" in advisories_str


def test_farmer_harvest_lodging_rain_alert():
    """Verify Farmer persona triggers HARVEST LODGING WARNING when crop is maturing and rain >= 10mm."""
    inp = PersonaInput(
        city="Patna",
        date="2026-11-05",
        temp_max=28.0,
        temp_min=18.0,
        apparent_temp_max=29.0,
        precipitation_mm=18.5,
        rain_probability=0.85,
        pm2_5=120.0,
        wind_speed_kmh=12.0,
        relative_humidity=85.0,
        solar_radiation=12.0,
        et0_evapotranspiration=3.0,
        crop_growth_stage="Kharif Rice: Harvesting Window",
        crop_harvest_dates="Oct 20 - Nov 20",
    )

    result = evaluate_farmer_persona(inp)
    restrictions_str = " ".join(result.operational_restrictions)
    assert "HARVEST LODGING WARNING" in restrictions_str
    assert "lodging and mold" in restrictions_str


def test_city_official_and_industry_thermal_capacity_integration():
    """Verify City Official and Industry personas ingest and display regional thermal power footprint."""
    inp = PersonaInput(
        city="Nagpur",
        date="2026-05-18",
        temp_max=44.0,
        temp_min=29.0,
        apparent_temp_max=46.0,
        precipitation_mm=0.0,
        rain_probability=0.0,
        pm2_5=165.0,
        wind_speed_kmh=7.0,
        relative_humidity=35.0,
        solar_radiation=26.0,
        et0_evapotranspiration=7.5,
        historical_percentile=97.0,
        atmospheric_stagnation=1,
        thermal_plants_within_100km=28,
        thermal_capacity_100km_mw=29092.0,
        closest_thermal_plant="KORADI TPP (2,400 MW, 12.4km N)",
    )

    # City Official test
    res_city = evaluate_city_official_persona(inp)
    assert "29,092 MW" in res_city.key_metrics["regional_thermal_power_capacity_100km"]
    assert any("Regional Airshed Alert" in a for a in res_city.advisories)

    # Industry test
    res_ind = evaluate_industry_persona(inp)
    assert "29,092 MW" in res_ind.key_metrics["regional_thermal_capacity_100km"]
    assert any("Point-Source Emission Confluence" in a for a in res_ind.advisories)
