"""Unit tests for Biometeorological Heat Intelligence Engine V1."""

import pytest

from climacore.analytics.biometeorology import (
    EstimatedWBGTResult,
    HeatIndexCategory,
    MultiNightHeatExposureResult,
    RothfuszHeatIndexResult,
    UTCICategory,
    UTCIResult,
    WBGTWorkRestCycle,
    calculate_estimated_wbgt,
    calculate_multi_night_heat_index,
    calculate_rothfusz_heat_index,
    calculate_utci,
    dispatch_heat_health_officer,
    dispatch_outdoor_worker_advisory,
)


def test_rothfusz_heat_index_basic():
    # 35°C, 60% RH should produce Dangerous Heat Index (~45-48°C apparent temp)
    res = calculate_rothfusz_heat_index(temp_c=35.0, rh_pct=60.0)
    assert res.heat_index_c > 35.0
    assert res.category in (HeatIndexCategory.DANGER, HeatIndexCategory.EXTREME_DANGER)
    assert res.heat_index_f > 100.0
    assert res.is_shade_assumed is True


def test_rothfusz_heat_index_normal_conditions():
    # 22°C, 50% RH is Normal
    res = calculate_rothfusz_heat_index(temp_c=22.0, rh_pct=50.0)
    assert res.category == HeatIndexCategory.NORMAL
    assert pytest.approx(res.heat_index_c, abs=1.5) == 22.0


def test_utci_calculation():
    # 35°C, 50% RH, moderate wind 2 m/s
    res = calculate_utci(temp_c=35.0, rh_pct=50.0, wind_speed_10m_ms=2.0)
    assert isinstance(res, UTCIResult)
    assert res.utci_c >= 28.0
    assert res.category in (
        UTCICategory.MODERATE_HEAT_STRESS,
        UTCICategory.STRONG_HEAT_STRESS,
        UTCICategory.VERY_STRONG_HEAT,
        UTCICategory.EXTREME_HEAT_STRESS,
    )

    # Moderate thermal comfort conditions: 20°C, 50% RH, 1 m/s wind
    res_comf = calculate_utci(temp_c=20.0, rh_pct=50.0, wind_speed_10m_ms=1.0)
    assert res_comf.category == UTCICategory.NO_THERMAL_STRESS


def test_estimated_wbgt_sun_vs_shade():
    # Under 800 W/m² direct solar radiation, outdoor WBGT must exceed shaded WBGT
    res = calculate_estimated_wbgt(
        temp_c=38.0,
        rh_pct=50.0,
        wind_speed_10m_ms=1.5,
        solar_radiation_wm2=800.0,
        is_outdoor_sun=True,
    )
    assert isinstance(res, EstimatedWBGTResult)
    assert res.estimated_wbgt_outdoor_c > res.estimated_wbgt_shaded_c
    assert res.is_instrument_measured is False
    assert "Not an instrument black-globe measurement" in res.disclaimer


def test_wbgt_work_rest_cycles_intensity_dependence():
    # In warm conditions, heavy work must require more rest than light work
    res = calculate_estimated_wbgt(
        temp_c=33.0,
        rh_pct=60.0,
        wind_speed_10m_ms=1.0,
        solar_radiation_wm2=600.0,
        is_outdoor_sun=True,
    )
    # Heavy work should have a more restrictive cycle than light work
    cycles = [
        WBGTWorkRestCycle.CONTINUOUS,
        WBGTWorkRestCycle.WORK_45_REST_15,
        WBGTWorkRestCycle.WORK_30_REST_30,
        WBGTWorkRestCycle.WORK_15_REST_45,
        WBGTWorkRestCycle.STOP_WORK_RECOMMENDED,
    ]
    light_idx = cycles.index(res.light_work_cycle)
    heavy_idx = cycles.index(res.heavy_work_cycle)
    assert heavy_idx >= light_idx


def test_multi_night_heat_index_recovery():
    # Series of 5 nights: 3 tropical nights (29°C, 28°C, 27°C) and 2 cool nights (22°C, 23°C)
    temps = [29.0, 28.0, 27.0, 22.0, 23.0]
    res = calculate_multi_night_heat_index(nightly_min_temps_c=temps)

    assert isinstance(res, MultiNightHeatExposureResult)
    assert res.tropical_nights_count == 3
    assert res.severe_tropical_nights_count == 2
    # Debt = (29 - 25) + (28 - 25) + (27 - 25) = 4 + 3 + 2 = 9.0 °C-days
    assert pytest.approx(res.cumulative_nocturnal_debt_deg_c, abs=0.1) == 9.0
    assert res.recovery_quality_score < 100.0
    assert res.prolonged_heat_warning is True
    assert "Does not constitute clinical or medical diagnosis" in res.clinical_disclaimer


def test_persona_dispatches():
    hi = calculate_rothfusz_heat_index(temp_c=42.0, rh_pct=40.0)
    utci = calculate_utci(temp_c=42.0, rh_pct=40.0)
    multi_night = calculate_multi_night_heat_index(nightly_min_temps_c=[29.0, 29.5, 30.0])

    officer_dispatch = dispatch_heat_health_officer(hi, utci, multi_night)
    assert officer_dispatch["persona"] == "HeatHealthOfficer"
    assert officer_dispatch["alert_level"] in ("ORANGE", "RED")
    assert len(officer_dispatch["recommended_actions"]) > 0

    wbgt = calculate_estimated_wbgt(temp_c=36.0, rh_pct=50.0, solar_radiation_wm2=700.0)
    worker_dispatch = dispatch_outdoor_worker_advisory(wbgt, work_intensity="heavy", is_acclimatized=False)
    assert worker_dispatch["persona"] == "OutdoorWorkerAdvisory"
    assert "UNACCLIMATIZED WORKER NOTICE" in worker_dispatch["guidance"][0]


def test_utci_domain_boundary_validation():
    # Valid domain test
    res_valid = calculate_utci(temp_c=30.0, rh_pct=50.0, wind_speed_10m_ms=2.0)
    assert res_valid.is_within_valid_domain is True
    assert len(res_valid.domain_boundary_warnings) == 0
    assert res_valid.mean_radiant_temp_assumed_shaded is True

    # Out of boundary test: wind speed 25 m/s (clipped to 17 m/s) and extreme temperature 55°C
    res_invalid = calculate_utci(temp_c=55.0, rh_pct=20.0, wind_speed_10m_ms=25.0)
    assert res_invalid.is_within_valid_domain is False
    assert len(res_invalid.domain_boundary_warnings) >= 1
    assert any("clipped" in w for w in res_invalid.domain_boundary_warnings)


def test_wbgt_metadata_and_stull_psychrometric_distinction():
    res = calculate_estimated_wbgt(
        temp_c=35.0,
        rh_pct=60.0,
        wind_speed_10m_ms=2.0,
        solar_radiation_wm2=700.0,
        is_outdoor_sun=True,
    )
    assert res.thermal_stress_index_name == "Estimated Thermal Stress Index (Simplified WBGT Proxy)"
    assert res.simplified_wbgt_outdoor_c == res.estimated_wbgt_outdoor_c
    assert res.simplified_wbgt_shaded_c == res.estimated_wbgt_shaded_c
    assert res.wet_bulb_method == "aspirated_psychrometric_stull2011"
    assert res.globe_temp_method == "liljegren_bom_empirical_approximation"
    assert res.is_instrument_measured is False


def test_multi_night_threshold_convention():
    res = calculate_multi_night_heat_index(nightly_min_temps_c=[26.0, 27.0, 29.0])
    assert "Project-Specific Regional Thresholds" in res.threshold_convention
    assert res.tropical_nights_count == 3
    assert res.severe_tropical_nights_count == 1
