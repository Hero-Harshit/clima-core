"""Unit tests for Hydrological Runoff Engine V1 (SCS Curve Number)."""

from pathlib import Path
import lightgbm as lgb
import pytest

from climacore.models.hydrology.scs_cn import (
    AntecedentMoistureCondition,
    HydrologicSoilGroup,
    LandCoverType,
    RunoffResult,
    SCSCurveNumberEngine,
    calculate_scs_runoff,
)


def test_amc_classification():
    engine = SCSCurveNumberEngine()

    # Growing season
    assert engine.classify_amc(10.0, is_growing_season=True) == AntecedentMoistureCondition.AMC_I
    assert engine.classify_amc(40.0, is_growing_season=True) == AntecedentMoistureCondition.AMC_II
    assert engine.classify_amc(60.0, is_growing_season=True) == AntecedentMoistureCondition.AMC_III

    # Dormant season
    assert engine.classify_amc(5.0, is_growing_season=False) == AntecedentMoistureCondition.AMC_I
    assert engine.classify_amc(20.0, is_growing_season=False) == AntecedentMoistureCondition.AMC_II
    assert engine.classify_amc(35.0, is_growing_season=False) == AntecedentMoistureCondition.AMC_III


def test_cn_amc_adjustment():
    engine = SCSCurveNumberEngine()
    cn2 = 80.0

    # AMC-II must equal base
    assert engine.adjust_cn_for_amc(cn2, AntecedentMoistureCondition.AMC_II) == 80.0

    # AMC-I must be lower than base
    cn1 = engine.adjust_cn_for_amc(cn2, AntecedentMoistureCondition.AMC_I)
    assert 15.0 <= cn1 < 80.0

    # AMC-III must be higher than base
    cn3 = engine.adjust_cn_for_amc(cn2, AntecedentMoistureCondition.AMC_III)
    assert 80.0 < cn3 <= 99.0


def test_zero_runoff_below_initial_abstraction():
    engine = SCSCurveNumberEngine()
    # CN = 75, S = 25400/75 - 254 = 84.67 mm, Ia = 0.2 * 84.67 = 16.93 mm
    res = engine.calculate_event_runoff(
        rainfall_mm=10.0,
        cn2=75.0,
        catchment_area_km2=2.0,
    )
    assert res.runoff_depth_mm == 0.0
    assert res.runoff_volume_m3 == 0.0
    assert res.runoff_coefficient == 0.0
    assert res.drainage_overflow_claim is False


def test_positive_runoff_above_initial_abstraction():
    engine = SCSCurveNumberEngine()
    res = engine.calculate_event_runoff(
        rainfall_mm=100.0,
        cn2=80.0,
        catchment_area_km2=5.0,  # 5 km²
    )
    assert res.runoff_depth_mm > 0.0
    assert res.runoff_depth_mm < 100.0
    assert 0.0 < res.runoff_coefficient < 1.0

    # Check volume conversion: V = Q (mm) * Area (km²) * 1000
    expected_volume = res.runoff_depth_mm * 5.0 * 1000.0
    assert pytest.approx(res.runoff_volume_m3, abs=1.0) == expected_volume


def test_connected_impervious_area_composite():
    engine = SCSCurveNumberEngine()
    # Pervious CN = 60, Impervious fraction = 0.50
    # Expected CN_composite = 0.5 * 60 + 0.5 * 98 = 79.0
    comp_cn = engine.compute_composite_cn(pervious_cn=60.0, connected_impervious_fraction=0.50)
    assert pytest.approx(comp_cn, abs=0.1) == 79.0

    res = engine.calculate_event_runoff(
        rainfall_mm=50.0,
        cn2=60.0,
        connected_impervious_fraction=0.50,
    )
    assert pytest.approx(res.base_curve_number_cn2, abs=0.1) == 79.0


def test_lambda_sensitivity():
    engine = SCSCurveNumberEngine()
    sens = engine.test_lambda_sensitivity(rainfall_mm=80.0, cn2=75.0)
    assert "lambda_0.20" in sens
    assert "lambda_0.05" in sens
    assert sens["lambda_0.20"].initial_abstraction_ratio == 0.20
    assert sens["lambda_0.05"].initial_abstraction_ratio == 0.05
    # Both must report positive runoff
    assert sens["lambda_0.20"].runoff_depth_mm > 0.0
    assert sens["lambda_0.05"].runoff_depth_mm > 0.0


def test_anti_hallucination_disclaimer():
    res = calculate_scs_runoff(rainfall_mm=120.0, cn=85.0)
    assert res.drainage_overflow_claim is False
    assert "No urban stormwater pipe surcharge or drain overflow is claimed" in res.disclaimer


def test_flood_risk_scs_challenger_artifact_loaded():
    challenger_path = Path("artifacts/models/flood_risk/v0.1.0/flood_risk_challenger_scs_cn.booster")
    assert challenger_path.exists()
    booster = lgb.Booster(model_file=str(challenger_path))
    assert booster.num_feature() == 23  # 20 base features + 3 SCS features
