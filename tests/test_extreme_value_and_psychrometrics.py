"""
Unit and integration tests for Phase 2: Scientific & Statistical Upgrades.
Validates:
1. GEV Extreme Value Theory return period calculations on 74y IMD Temp and 46y IMD Rain.
2. Elevation and pressure-aware psychrometric thermodynamics.
3. Transparent empirical multi-hazard joint tail exceedance index.
4. Civil defense and actuarial persona integrations.
"""

import json
from pathlib import Path
import pytest
from climacore.config.loader import get_project_root
from climacore.analytics.extreme_value import (
    load_gev_profiles,
    evaluate_return_period,
    RETURN_PERIODS,
)
from climacore.analytics.psychrometrics import (
    estimate_station_pressure_hpa,
    calculate_elevation_aware_wet_bulb,
    get_city_elevation,
)
from climacore.models.compound_risk.scoring import CompoundRiskEngine
from climacore.personas.models import PersonaInput
from climacore.personas.city_official import evaluate_city_official_persona
from climacore.personas.insurance import evaluate_insurance_persona
from climacore.personas.industry import evaluate_industry_persona


@pytest.fixture(scope="module")
def project_root() -> Path:
    return get_project_root()


def test_gev_profiles_coverage_and_monotonicity(project_root):
    """Verify city_gev_return_periods.json covers 50 cities and return levels are strictly monotonic."""
    profiles = load_gev_profiles()
    assert len(profiles) >= 50, f"Expected 50 city GEV profiles, got {len(profiles)}"

    for city, prof in profiles.items():
        assert "temperature_gev" in prof, f"Missing temperature GEV for {city}"
        assert "rainfall_gev" in prof, f"Missing rainfall GEV for {city}"

        t_ret = prof["temperature_gev"]["return_levels"]
        r_ret = prof["rainfall_gev"]["return_levels"]

        # Strictly monotonic return levels: T_2yr <= T_5yr <= T_10yr <= T_25yr <= T_50yr
        assert t_ret["T_2yr"] <= t_ret["T_5yr"] <= t_ret["T_10yr"] <= t_ret["T_25yr"] <= t_ret["T_50yr"], (
            f"Non-monotonic temperature return levels in {city}: {t_ret}"
        )
        assert r_ret["T_2yr"] <= r_ret["T_5yr"] <= r_ret["T_10yr"] <= r_ret["T_25yr"] <= r_ret["T_50yr"], (
            f"Non-monotonic rainfall return levels in {city}: {r_ret}"
        )


def test_gev_manifest_integrity(project_root):
    """Verify cryptographic manifest exists with valid SHA-256."""
    manifest_path = project_root / "data" / "manifests" / "gev_return_periods_manifest.json"
    assert manifest_path.exists(), f"Missing GEV manifest at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "Extreme Value Theory" in manifest["dataset_name"]
    assert manifest["total_cities"] >= 50
    assert len(manifest["sha256"]) == 64


def test_evaluate_return_period_accuracy():
    """Verify evaluate_return_period properly maps rare events to high return periods."""
    # Delhi 47.0°C should be a severe/very rare event (T >= 10 years)
    delhi_heat = evaluate_return_period("Delhi", 47.0, hazard_type="temperature")
    assert delhi_heat["estimated_return_period_years"] >= 10.0
    assert "EXTREME" in delhi_heat["rarity_category"] or "ANOMALY" in delhi_heat["rarity_category"]

    # Bengaluru 130mm rainfall should be a very rare pluvial event
    blr_rain = evaluate_return_period("Bengaluru", 130.0, hazard_type="rainfall")
    assert blr_rain["estimated_return_period_years"] >= 10.0
    assert "SEVERE" in blr_rain["rarity_category"] or "EXTREME" in blr_rain["rarity_category"]

    # Mild day (Delhi 32°C) should be typical (< 2 years)
    mild_heat = evaluate_return_period("Delhi", 32.0, hazard_type="temperature")
    assert mild_heat["estimated_return_period_years"] <= 2.0
    assert "TYPICAL" in mild_heat["rarity_category"]


def test_elevation_aware_psychrometrics():
    """Verify barometric pressure and elevation cooling effect on wet-bulb temperature."""
    # Bangalore is at 920m -> station pressure should be ~905-915 hPa
    p_blr = estimate_station_pressure_hpa(920.0)
    assert 905.0 <= p_blr <= 915.0

    # At 38°C and 60% RH:
    # Bangalore (elev 920m) should experience enhanced evaporative cooling
    # leading to a cooler wet-bulb temperature than sea-level Stull formula
    blr_wb = calculate_elevation_aware_wet_bulb(temp_c=38.0, rh_pct=60.0, city="Bengaluru")
    mumbai_wb = calculate_elevation_aware_wet_bulb(temp_c=38.0, rh_pct=60.0, city="Mumbai")

    assert blr_wb["wet_bulb_temp_c"] < blr_wb["sea_level_stull_wet_bulb_c"]
    assert blr_wb["station_pressure_hpa"] < mumbai_wb["station_pressure_hpa"]
    assert blr_wb["wet_bulb_temp_c"] <= 38.0  # Must not exceed dry bulb


def test_compound_risk_joint_tail_exceedance():
    """Verify transparent multi-hazard joint tail exceedance quantification."""
    engine = CompoundRiskEngine()
    res = engine.calculate_single(
        pm2_5_pred=280.0,     # Severe Air Quality (~100 score)
        temp_max_pred=44.0,   # Severe Heat (~90 score)
        precip_pred=5.0,      # Low rain
        city="Delhi",
        date="2026-06-10",
    )

    assert res["is_compounded_event"] is True
    assert "joint_tail_exceedance" in res
    jt = res["joint_tail_exceedance"]
    assert "bivariate_probabilities" in jt
    assert jt["bivariate_probabilities"]["heat_and_air_pollution"] > 0.70
    assert jt["trivariate_probability"] < jt["bivariate_probabilities"]["heat_and_air_pollution"]


def test_city_official_evt_return_period_integration():
    """Verify City Official persona displays EVT return period and declares GEV extreme restrictions."""
    inp = PersonaInput(
        city="Delhi",
        date="2026-06-15",
        temp_max=46.8,       # ~25-year GEV return level in Delhi
        temp_min=29.0,
        apparent_temp_max=49.0,
        precipitation_mm=0.0,
        rain_probability=0.0,
        pm2_5=85.0,
        wind_speed_kmh=12.0,
        relative_humidity=40.0,
        solar_radiation=24.0,
        et0_evapotranspiration=6.5,
        historical_percentile=98.5,
    )

    result = evaluate_city_official_persona(inp)
    assert "extreme_value_gev_return_period" in result.key_metrics
    assert any("EVT" in r for r in result.operational_restrictions or result.advisories)


def test_insurance_evt_parametric_trigger_integration():
    """Verify Insurance persona fires 1-in-10 year parametric strike on GEV exceedance."""
    inp = PersonaInput(
        city="Bengaluru",
        date="2026-09-20",
        temp_max=28.0,
        temp_min=19.0,
        apparent_temp_max=29.0,
        precipitation_mm=135.0,  # > 1-in-10 year GEV pluvial return level in Bengaluru
        rain_probability=0.95,
        pm2_5=40.0,
        wind_speed_kmh=14.0,
        relative_humidity=85.0,
        solar_radiation=14.0,
        et0_evapotranspiration=3.2,
    )

    result = evaluate_insurance_persona(inp)
    assert "PARAMETRIC_FLOOD_STRIKE" in result.key_metrics["pluvial_flood_exposure"]
    assert "evt_rainfall_return_period" in result.key_metrics
    assert float(result.key_metrics["evt_rainfall_return_period"].split()[0]) >= 10.0


def test_industry_elevation_aware_wet_bulb_integration():
    """Verify Industry persona includes pressure-adjusted wet bulb and station elevation."""
    inp = PersonaInput(
        city="Pune",
        date="2026-05-10",
        temp_max=41.0,
        temp_min=24.0,
        apparent_temp_max=42.0,
        precipitation_mm=0.0,
        rain_probability=0.0,
        pm2_5=95.0,
        wind_speed_kmh=10.0,
        relative_humidity=45.0,
        solar_radiation=24.0,
        et0_evapotranspiration=6.0,
    )

    result = evaluate_industry_persona(inp)
    assert "station_pressure" in result.key_metrics
    assert "560m" in result.key_metrics["station_pressure"]
