"""
Unit test suite for the 5 Flagship Champion Models (Phase 3 Consolidation).
Covers:
  1. FloodRisk V2 (decoupled pure hydrological hazard vs exposure impact)
  2. HeatRisk V3 (day/night dual model, UHI delta, GEV 74-year return level, IMD alerts)
  3. PowerGrid V2 (peak load MW, unmet supply stress, spinning reserves)
  4. AgriSense V2 (10-crop suitability, SAGE phenology, anthesis heat shock, harvest rain)
  5. AirQuality V2 (CPCB NAQI sub-indices, inversion stagnation, transboundary smoke)
  6. LiveForecastEngine flagship intelligence integration
"""

import pytest
from climacore.models.flood_risk.engine import FloodRiskEngine
from climacore.models.heat_risk.engine import HeatRiskEngine
from climacore.models.electricity_grid.engine import PowerGridEngine
from climacore.models.crop_resilience.engine import AgriSenseEngine
from climacore.models.air_quality.engine import (
    AirQualityEngine,
    compute_sub_index,
    classify_naqi,
    PM25_BREAKPOINTS,
    PM10_BREAKPOINTS,
    NO2_BREAKPOINTS,
)
from climacore.ingestion.engine import LiveForecastEngine


def test_flood_risk_engine_decoupling():
    """Verify FloodRisk strictly decouples pure hazard probability from exposure impact."""
    engine = FloodRiskEngine()

    # Dry scenario
    res_dry = engine.evaluate_flood_risk(
        city="Delhi",
        rainfall_mm=0.0,
        drainage_canals_15km=60,
        population_15km=5_000_000,
    )
    assert res_dry["hydrological_hazard_probability"] <= 0.05
    assert res_dry["hazard_tier"] == "NORMAL"

    # Extreme monsoon downpour scenario
    res_wet = engine.evaluate_flood_risk(
        city="Mumbai",
        rainfall_mm=120.0,
        rain_rolling_3d=180.0,
        drainage_canals_15km=12,  # bottleneck
        population_15km=12_000_000,
        hospitals_15km=45,
        schools_15km=180,
    )
    assert res_wet["hydrological_hazard_probability"] > 0.50
    assert res_wet["hazard_tier"] in ["HIGH_ALERT", "CRITICAL_DISASTER"]
    assert "exposure_footprint" in res_wet
    assert res_wet["exposure_footprint"]["population_15km"] == 12_000_000
    assert res_wet["exposure_footprint"]["hospitals_15km"] == 45
    assert len(res_wet["advisories"]) > 0


def test_heat_risk_engine_dual_and_gev():
    """Verify HeatRisk evaluates daytime, nighttime tropical failure, UHI, and GEV return levels."""
    engine = HeatRiskEngine()

    # Severe heatwave day + tropical night
    res = engine.evaluate_heat_risk(
        city="Delhi",
        temp_max_c=46.5,
        temp_min_c=31.0,
        relative_humidity=35.0,
        dense_cluster_pop=5_000_000,
    )
    assert res["daytime"]["temp_max_c"] == 46.5
    assert res["nighttime"]["tropical_night_failure"] is True
    assert res["nighttime"]["transformer_cooling_impaired"] is True
    assert res["uhi_microclimate"]["uhi_excess_c"] > 0.0
    assert "Red Alert" in res["imd_heatwave"]["alert_tier"] or "Orange Alert" in res["imd_heatwave"]["alert_tier"]
    assert "gev_return_period" in res
    assert len(res["advisories"]) > 0


def test_power_grid_engine_peak_and_stress():
    """Verify PowerGrid estimates peak load MW and unmet supply stress probability."""
    engine = PowerGridEngine()

    # High thermal load in Maharashtra
    res = engine.evaluate_power_grid_risk(
        temp_max_c=44.0,
        temp_min_c=29.0,
        state="Maharashtra",
        population_15km=8_000_000,
        thermal_plants_within_100km=4,
        thermal_capacity_100km_mw=5400.0,
    )
    assert res["cooling_degree_days_18c"] == pytest.approx(18.5, 0.5)
    assert res["cooling_demand_surge_ratio"] > 1.20
    assert res["peak_load_mw"] > 10000.0
    assert 0.0 <= res["unmet_supply_stress_probability"] <= 1.0
    assert res["discom_advisory"]["status"] in ["ELEVATED_STRAIN", "CRITICAL_BLACKOUT_RISK"]


def test_agrisense_engine_suitability_and_phenology():
    """Verify AgriSense multi-crop suitability, SAGE phenology, and thermal/rainfall stress."""
    engine = AgriSenseEngine()

    # Agartala in early May (DOY 125, Kharif Anthesis window)
    res_anthesis = engine.evaluate_agrisense(
        city="Agartala",
        doy=125,
        temp_max_c=37.0,  # Anthesis heat shock
        temp_min_c=24.0,
        temp_mean_c=30.5,
        rainfall_mm=2.0,
        humidity_pct=65.0,
        et0_evapotranspiration=5.2,
    )
    assert res_anthesis["phenology"]["is_anthesis"] is True
    assert res_anthesis["phenological_risks"]["anthesis_heat_stress"] is True
    assert len(res_anthesis["top_suitable_crops"]) >= 3
    assert res_anthesis["top_suitable_crops"][0]["probability"] > 0.0
    assert res_anthesis["physical_water_balance"]["water_balance_mm_day"] < 0.0  # deficit

    # Agra in harvest window with heavy rain (lodging threat)
    res_harvest = engine.evaluate_agrisense(
        city="Agra",
        doy=315,  # Kharif Bajra harvest window (doy 302-329)
        temp_max_c=28.0,
        temp_min_c=16.0,
        temp_mean_c=22.0,
        rainfall_mm=25.0,  # Rain during harvest
        humidity_pct=85.0,
        et0_evapotranspiration=3.0,
    )
    assert res_harvest["phenology"]["is_harvest"] is True
    assert res_harvest["phenological_risks"]["harvest_lodging_risk"] is True
    assert any("HARVEST" in a or "Harvest" in a for a in res_harvest["advisories"] + res_harvest["restrictions"])


def test_air_quality_engine_naqi_and_stagnation():
    """Verify AirQuality sub-index calculation, NAQI categorisation, and inversion trapping."""
    engine = AirQualityEngine()

    # CPCB sub-index math
    sub_55 = compute_sub_index(55.0, PM25_BREAKPOINTS)
    assert 51.0 <= sub_55 <= 100.0  # Satisfactory band

    cat, hex_color = classify_naqi(350.0)
    assert cat == "Very Poor"

    # Delhi in November: calm winds, high particulate, stubble season
    res_delhi = engine.evaluate_air_quality(
        city="Delhi",
        pm2_5_ug_m3=240.0,
        pm10_ug_m3=390.0,
        no2_ug_m3=65.0,
        wind_speed_kmh=6.0,
        temp_max_c=26.0,
        temp_min_c=12.0,
        precipitation_mm=0.0,
        month=11,
        day_of_year=310,
        population_15km=18_000_000,
        schools_15km=450,
        hospitals_15km=120,
    )
    assert res_delhi["naqi"] >= 300.0
    assert res_delhi["naqi_category"] in ["Very Poor", "Severe"]
    assert res_delhi["atmospheric_physics"]["stagnation_flag"] == 1
    assert res_delhi["atmospheric_physics"]["transboundary_smoke_risk"] is True
    assert any("GRAP" in r or "mask" in a.lower() for r in res_delhi["restrictions"] for a in res_delhi["advisories"] + [r])


def test_live_forecast_engine_flagship_integration():
    """Verify LiveForecastEngine executes all 5 Flagship Engines and exposes flagship_intelligence."""
    engine = LiveForecastEngine()
    result = engine.predict(query="Delhi", persona="all", days=3)

    assert result.city == "Delhi"
    assert len(result.daily_predictions) == 3

    day0 = result.daily_predictions[0]
    assert day0.flagship_intelligence is not None
    fi = day0.flagship_intelligence

    # Check that all 5 Flagship outputs are populated
    assert "flood_risk" in fi
    assert "hydrological_hazard_probability" in fi["flood_risk"]

    assert "heat_risk" in fi
    assert "daytime" in fi["heat_risk"]
    assert "nighttime" in fi["heat_risk"]

    assert "power_grid" in fi
    assert "peak_load_mw" in fi["power_grid"]
    assert "cooling_demand_surge_ratio" in fi["power_grid"]

    assert "agrisense" in fi
    assert "phenology" in fi["agrisense"]
    assert "top_suitable_crops" in fi["agrisense"]

    assert "air_quality" in fi
    assert "naqi" in fi["air_quality"]
    assert "atmospheric_physics" in fi["air_quality"]

    # Verify persona dict still works seamlessly
    assert len(day0.personas) == 13
