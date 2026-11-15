"""
Unit tests for the CompoundRisk engine.
Verifies normalization curves, weight validation, multi-hazard amplification,
and driver attribution logic.
"""

import pytest
from climacore.models.compound_risk.config import (
    HazardWeights,
    normalize_air_quality,
    normalize_heat_risk,
    normalize_rain_risk,
    classify_compound_risk_category,
)
from climacore.models.compound_risk.scoring import CompoundRiskEngine


def test_normalization_bounds():
    # Air quality
    score_clean, cat_clean = normalize_air_quality(15.0)
    score_severe, cat_severe = normalize_air_quality(350.0)
    assert 0.0 <= score_clean <= 20.0
    assert cat_clean == "Good"
    assert 90.0 <= score_severe <= 100.0
    assert cat_severe == "Severe"

    # Heat risk
    score_cool, cat_cool = normalize_heat_risk(22.0)
    score_hot, cat_hot = normalize_heat_risk(45.5)
    assert 0.0 <= score_cool <= 20.0
    assert 80.0 <= score_hot <= 95.0
    assert cat_hot == "Heatwave"

    # Rain risk
    score_dry, cat_dry = normalize_rain_risk(0.0)
    score_heavy, cat_heavy = normalize_rain_risk(85.0)
    assert score_dry == 0.0
    assert 65.0 <= score_heavy <= 85.0
    assert cat_heavy == "Heavy Rain"


def test_weights_validation():
    valid = HazardWeights(air_quality=0.4, heat_risk=0.4, rain_risk=0.2)
    valid.validate_weights()

    invalid = HazardWeights(air_quality=0.5, heat_risk=0.5, rain_risk=0.5)
    with pytest.raises(ValueError, match="Hazard weights must sum to 1.0"):
        invalid.validate_weights()


def test_compounding_multiplier_behavior():
    engine = CompoundRiskEngine(compound_amplification_factor=0.25, elevated_threshold=50.0)

    # Case 1: Only one elevated hazard (High heat 42C -> score ~75, but clean air and dry)
    res_single = engine.calculate_single(pm2_5_pred=15.0, temp_max_pred=42.0, precip_pred=0.0)
    assert not res_single["is_compounded_event"]
    assert res_single["compound_multiplier"] == 1.0

    # Case 2: Co-occurring compound hazards (High heat 42C + Very Poor PM2.5 150 ug/m3)
    res_compound = engine.calculate_single(pm2_5_pred=150.0, temp_max_pred=42.0, precip_pred=0.0)
    assert res_compound["is_compounded_event"]
    assert res_compound["compound_multiplier"] > 1.0
    assert res_compound["compound_risk_score"] > res_single["compound_risk_score"]


def test_dominant_driver_attribution():
    engine = CompoundRiskEngine()
    # High PM2.5 in winter Delhi (low heat, zero rain)
    res_smog = engine.calculate_single(pm2_5_pred=280.0, temp_max_pred=18.0, precip_pred=0.0)
    assert res_smog["dominant_driver"] == "Air Quality (PM2.5)"
    assert res_smog["sub_hazards"]["air_quality"]["contribution_pct"] > 60.0

    # Monsoon deluge in Mumbai (moderate heat, clean rain-washed air)
    res_deluge = engine.calculate_single(pm2_5_pred=12.0, temp_max_pred=29.0, precip_pred=120.0)
    assert res_deluge["dominant_driver"] == "Rainfall Hazard"
    assert res_deluge["sub_hazards"]["rain_risk"]["contribution_pct"] > 60.0
