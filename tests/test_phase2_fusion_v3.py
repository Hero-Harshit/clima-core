"""
Unit and Integration Test Suite for Phase 2 · Cross-Dataset Fusion & Flagship V3 Models.
Validates:
  1. FloodRisk V3: Hydro-atmospheric cascade (soil saturation, effective runoff, drainage bottleneck, multi-horizon 24h/48h/72h).
  2. HeatRisk V4: Dual diurnal cooling deficit, Oke UHI microclimate excess, quantile bounds (Q10, Q50, Q90), multi-horizon.
  3. PowerGrid V3: Historical load lags (1d, 7d, 14d), weekend/holiday calendar indicators, quantile load bounds, multi-horizon.
  4. AgriSense V3: Tri-task architecture (suitability GBDT, yield potential quantiles in t/ha, SAGE phenology stress).
  5. AirQuality V3: Ventilation index (WS * BLH), nocturnal inversion trapping, CAMS advection coupling, multi-horizon (24h/48h/72h).
"""

import pytest
from climacore.models.flood_risk.engine import FloodRiskEngine
from climacore.models.heat_risk.engine import HeatRiskEngine
from climacore.models.electricity_grid.engine import PowerGridEngine
from climacore.models.crop_resilience.engine import AgriSenseEngine
from climacore.models.air_quality.engine import AirQualityEngine
from climacore.ingestion.engine import LiveForecastEngine


def test_flood_risk_v3_hydro_atmospheric_cascade():
    """Verify FloodRisk V3 hydro-atmospheric cascade: soil saturation, runoff, and multi-horizon probabilities."""
    engine = FloodRiskEngine()

    # Dry unsaturated topsoil scenario
    dry = engine.evaluate_flood_risk(
        city="Jaipur",
        rainfall_mm=10.0,
        rootzone_saturation_pct=30.0,
        drainage_canals_15km=60,
        arterial_roads_15km=100,
    )
    assert dry["hydro_cascade"]["infiltration_state"] == "NORMAL_SOIL_INFILTRATION"
    assert dry["hydro_cascade"]["runoff_coefficient"] <= 0.25
    assert dry["multi_horizon_hazard"]["p_24h_flash_inundation"] <= 0.20

    # Saturated topsoil + drainage bottleneck scenario
    saturated = engine.evaluate_flood_risk(
        city="Mumbai",
        rainfall_mm=110.0,
        rain_rolling_3d=180.0,
        rain_rolling_7d=280.0,
        rootzone_saturation_pct=85.0,  # >70% collapses infiltration
        drainage_canals_15km=15,       # bottleneck
        arterial_roads_15km=120,
        population_15km=12_000_000,
        dense_cluster_pop_15km=2_500_000,
    )
    assert saturated["hydro_cascade"]["infiltration_state"] == "SATURATED_OVERLAND_FLOW"
    assert saturated["hydro_cascade"]["runoff_coefficient"] >= 0.50
    assert saturated["hydro_cascade"]["drainage_impedance_factor"] > 1.30
    assert saturated["multi_horizon_hazard"]["p_24h_flash_inundation"] > 0.50
    assert saturated["multi_horizon_hazard"]["p_48h_channel_overflow"] > 0.50
    assert saturated["multi_horizon_hazard"]["p_72h_catchment_waterlogging"] > 0.50
    assert saturated["hazard_tier"] in ["HIGH_ALERT", "CRITICAL_DISASTER"]

    # Verify decoupling: exposure variations do not change physical hazard probability
    decoupled_variant = engine.evaluate_flood_risk(
        city="Mumbai",
        rainfall_mm=110.0,
        rain_rolling_3d=180.0,
        rain_rolling_7d=280.0,
        rootzone_saturation_pct=85.0,
        drainage_canals_15km=15,
        arterial_roads_15km=120,
        population_15km=200_000,  # 60x smaller population
        dense_cluster_pop_15km=10_000,
    )
    assert decoupled_variant["hydrological_hazard_probability"] == saturated["hydrological_hazard_probability"]
    assert decoupled_variant["impact_severity_index"] < saturated["impact_severity_index"]


def test_heat_risk_v4_diurnal_and_quantiles():
    """Verify HeatRisk V4 quantile uncertainty, nocturnal cooling deficit, and Oke UHI microclimate excess."""
    engine = HeatRiskEngine()

    res = engine.evaluate_heat_risk(
        city="Nagpur",
        temp_max_c=45.5,
        temp_min_c=30.0,  # Tropical night
        relative_humidity=38.0,
        wind_speed_kmh=5.0,  # Calm wind
        solar_radiation_mj=26.0,
        dense_cluster_pop=1_200_000,
    )

    # Quantile ordering checks
    q_day = res["daytime"]["quantiles"]
    assert q_day["q10"] < q_day["q50"] < q_day["q90"]
    assert q_day["q50"] == 45.5

    q_night = res["nighttime"]["quantiles"]
    assert q_night["q10"] < q_night["q50"] < q_night["q90"]
    assert q_night["q50"] == 30.0

    # Tropical night & cooling deficit
    assert res["is_tropical_night"] is True
    assert res["cooling_deficit_c"] >= 6.0  # 30.0 - 24.0

    # Oke UHI excess
    assert res["uhi_temperature_delta_c"] >= 2.0
    assert "multi_horizon_diurnal" in res
    assert "h24" in res["multi_horizon_diurnal"]
    assert "h48" in res["multi_horizon_diurnal"]
    assert "h72" in res["multi_horizon_diurnal"]


def test_power_grid_v3_lags_and_calendar():
    """Verify PowerGrid V3 load lags, weekend reduction factor, and quantile load bounds."""
    engine = PowerGridEngine()

    # Weekday baseline
    res_weekday = engine.evaluate_grid_stress(
        city="Ahmedabad",
        state="Gujarat",
        temp_max_c=43.0,
        temp_min_c=29.0,
        population_15km=7_000_000,
        load_lag_1d_mw=14_000.0,
        load_lag_7d_mw=13_800.0,
        load_lag_14d_mw=13_500.0,
        is_weekend=False,
    )

    # Weekend baseline with same weather and lags
    res_weekend = engine.evaluate_grid_stress(
        city="Ahmedabad",
        state="Gujarat",
        temp_max_c=43.0,
        temp_min_c=29.0,
        population_15km=7_000_000,
        load_lag_1d_mw=14_000.0,
        load_lag_7d_mw=13_800.0,
        load_lag_14d_mw=13_500.0,
        is_weekend=True,
    )

    assert res_weekend["peak_load_mw"] < res_weekday["peak_load_mw"]
    assert res_weekend["cooling_demand_surge_ratio"] < res_weekday["cooling_demand_surge_ratio"]

    # Quantile bounds
    q_load = res_weekday["quantiles_peak_load_mw"]
    assert q_load["q10"] < q_load["q50"] < q_load["q90"]
    assert q_load["q50"] == res_weekday["peak_load_mw"]

    # Multi-horizon
    assert "multi_horizon_grid" in res_weekday
    assert res_weekday["multi_horizon_grid"]["h24"]["peak_load_mw"] > 0.0


def test_agrisense_v3_tri_task_architecture():
    """Verify AgriSense V3 tri-task output: suitability, yield potential quantiles, and SAGE phenology."""
    engine = AgriSenseEngine()

    # Agartala in anthesis window with heat shock
    res = engine.evaluate_agrisense(
        city="Agartala",
        doy=125,
        temp_max_c=36.5,  # Heat shock during anthesis
        temp_min_c=25.0,
        temp_mean_c=30.0,
        rainfall_mm=2.0,
        humidity_pct=60.0,
        smdi=-1.8,  # Severe drought
    )

    # Task A: Suitability
    assert len(res["top_suitable_crops"]) >= 3
    assert res["top_suitable_crops"][0]["probability"] > 0.0

    # Task B: Yield Potential Quantiles
    yield_q = res["yield_potential_quantiles"]
    assert yield_q["q10_yield_t_ha"] < yield_q["q50_yield_t_ha"] < yield_q["q90_yield_t_ha"]
    assert yield_q["stress_penalty_factor"] < 1.0  # penalized due to anthesis + drought
    assert yield_q["unit"] == "tonnes/hectare"

    # Task C: Phenological Risk
    assert res["phenology"]["is_anthesis"] is True
    assert res["phenological_risks"]["anthesis_heat_stress"] is True
    assert res["phenological_risks"]["anthesis_severity"] == "SEVERE_FLORET_STERILITY"

    # Multi-horizon stress trajectory
    assert "multi_horizon_agro_stress" in res
    assert "h24" in res["multi_horizon_agro_stress"]
    assert "h48" in res["multi_horizon_agro_stress"]
    assert "h72" in res["multi_horizon_agro_stress"]


def test_air_quality_v3_ventilation_and_advection():
    """Verify AirQuality V3 ventilation index (WS * BLH), inversion trapping, and multi-horizon multi-pollutant forecasts."""
    engine = AirQualityEngine()

    # Winter nocturnal inversion in Delhi with low ventilation
    res = engine.evaluate_air_quality(
        city="Delhi",
        pm2_5_ug_m3=280.0,
        pm10_ug_m3=450.0,
        no2_ug_m3=75.0,
        wind_speed_kmh=4.0,  # ~1.1 m/s
        temp_max_c=25.0,
        temp_min_c=10.0,     # DTR = 15°C
        precipitation_mm=0.0,
        boundary_layer_height_m=280.0,
        month=11,
        day_of_year=315,
    )

    # Atmospheric physics checks
    phys = res["atmospheric_physics"]
    assert phys["ventilation_index_m2_s"] < 1500.0  # Severe pollutant trapping (<2000 m2/s)
    assert phys["stagnation_flag"] == 1
    assert phys["transboundary_smoke_risk"] is True

    # Multi-horizon projections
    mh = res["multi_horizon_air_quality"]
    assert "h24" in mh and "h48" in mh and "h72" in mh
    # Stagnation causes pollutant concentrations to accumulate over 48h and 72h
    assert mh["h48"]["pm2_5_ug_m3"] >= mh["h24"]["pm2_5_ug_m3"]
    assert mh["h72"]["pm2_5_ug_m3"] >= mh["h48"]["pm2_5_ug_m3"]

    # CPCB 8-pollutant compliance
    comp = res["cpcb_compliance"]
    assert comp["is_cpcb_compliant"] is True
    assert comp["monitored_count"] >= 3
    assert comp["has_mandatory_pm"] is True


def test_live_forecast_engine_fusion_v3_integration():
    """Verify LiveForecastEngine end-to-end integration with the updated V3/V4 flagship engines."""
    engine = LiveForecastEngine()
    result = engine.predict(query="Mumbai", persona="all", days=3)

    assert result.city == "Mumbai"
    assert len(result.daily_predictions) == 3

    day0 = result.daily_predictions[0]
    fi = day0.flagship_intelligence

    # Check FloodRisk V3 additions
    assert "multi_horizon_hazard" in fi["flood_risk"]
    assert "hydro_cascade" in fi["flood_risk"]

    # Check HeatRisk V4 additions
    assert "multi_horizon_diurnal" in fi["heat_risk"]
    assert "quantiles" in fi["heat_risk"]["daytime"]
    assert "quantiles" in fi["heat_risk"]["nighttime"]

    # Check PowerGrid V3 additions
    assert "quantiles_peak_load_mw" in fi["power_grid"]
    assert "multi_horizon_grid" in fi["power_grid"]
    assert "lags_and_calendar" in fi["power_grid"]

    # Check AgriSense V3 additions
    assert "yield_potential_quantiles" in fi["agrisense"]
    assert "multi_horizon_agro_stress" in fi["agrisense"]

    # Check AirQuality V3 additions
    assert "ventilation_index_m2_s" in fi["air_quality"]["atmospheric_physics"]
    assert "multi_horizon_air_quality" in fi["air_quality"]
