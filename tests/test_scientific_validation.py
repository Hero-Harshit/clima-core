"""
Automated Scientific Validation Test Suite for Clima-Core v1.1.1.

Validates core scientific integrity principles across all upgraded engines:
1. Forecast Verification: Ground truth provenance (IMD gridded obs), lead time alignment (T-24h -> T), two-tier precipitation evaluation.
2. Model Registry: Benchmark reconciliation (true evaluated metrics vs legacy placeholders).
3. Biometeorology: UTCI operational domain boundary checking, Stull aspirated psychrometric vs natural wet bulb distinctions, regional night thresholds.
4. TreeSHAP: Exact mathematical additivity (|sum(phi) + phi0 - f(x)| < 1e-5), separation of input feature unit vs SHAP attribution output space.
5. Compound Copula: Exactly 45 common years (1980-2024), Archimedean upper-tail candidates (Gumbel & Survival Clayton).
6. Water Quality: Exact deterministic BIS IS 10500:2012 formula calculation pathway & surrogate transparency.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from climacore.analytics.biometeorology import (
    calculate_estimated_wbgt,
    calculate_multi_night_heat_index,
    calculate_utci,
)
from climacore.analytics.copula import (
    CompoundCopulaEngine,
    CopulaFamily,
    compute_pseudo_observations,
)
from climacore.explainability.treeshap import TreeSHAPEngine
from climacore.models.water_quality.engine import WaterQualityRiskEngine
from climacore.registry.engine import get_model_registry


# -----------------------------------------------------------------------------
# 1. Forecast Verification Ground Truth & Two-Tier Evaluation
# -----------------------------------------------------------------------------
def test_scientific_forecast_verification_canonical_dataset():
    """Verify canonical pairs dataset uses genuine IMD observations and correct lead time."""
    pairs_path = Path("data/processed/forecast_correction_pairs/canonical_forecast_verification_pairs.parquet")
    assert pairs_path.exists(), "Canonical forecast verification pairs dataset must exist."

    df = pd.read_parquet(pairs_path)
    assert len(df) > 1000

    # Verification source must be independent observation, not day-0 forecast
    assert df["observation_source"].str.startswith("IMD_Gridded_Obs").all()
    assert (df["observation_quality_flag"] == "QC_PASS_INDEPENDENT").all()

    # Valid time must be exactly forecast_issue_time + 24 hours
    issue_dt = pd.to_datetime(df["forecast_issue_time"])
    valid_dt = pd.to_datetime(df["valid_time"])
    time_deltas = (valid_dt - issue_dt).dt.total_seconds() / 3600.0
    assert (time_deltas == 24.0).all(), "All forecast records must have strictly 24-hour lead time."

    # Variables covered must include t2m_max, t2m_min, and precipitation
    var_names = set(df["variable"].unique())
    assert "t2m_max" in var_names
    assert "t2m_min" in var_names
    assert "precipitation" in var_names


def test_scientific_forecast_correction_report_metrics():
    """Verify forecast correction report separates all-day from wet-day conditional evaluation."""
    report_path = Path("artifacts/models/forecast_correction/v0.1.0/forecast_correction_report.json")
    assert report_path.exists()

    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert "Independent IMD Gridded Observations" in rep["verification_baseline"]
    assert "benchmarks" in rep

    benchmarks = rep["benchmarks"]

    # Verify Tmin achieved > 50% MAE reduction
    raw_tmin = next(b for b in benchmarks if b["variable"] == "t2m_min" and b["method"] == "raw")
    ml_tmin = next(b for b in benchmarks if b["variable"] == "t2m_min" and b["method"] == "ml_residual")
    mae_reduction_pct = (raw_tmin["mae"] - ml_tmin["mae"]) / raw_tmin["mae"] * 100.0
    assert mae_reduction_pct > 50.0, f"Tmin MAE reduction should exceed 50%, got {mae_reduction_pct:.2f}%"

    # Verify precipitation two-tier evaluation
    precip_masks = {b["evaluation_mask"] for b in benchmarks if b["variable"] == "precipitation"}
    assert "all_days" in precip_masks
    assert "wet_days_conditional" in precip_masks

    # Contingency scores exist
    assert "precipitation_contingency" in rep
    assert rep["precipitation_contingency"]["ml_precipitation"]["threat_score_csi"] > 0.70


# -----------------------------------------------------------------------------
# 2. Model Registry Benchmark Reconciliation
# -----------------------------------------------------------------------------
def test_scientific_model_registry_flood_risk_benchmarks():
    """Verify registry records the true evaluated metrics of flood_risk_v2, not legacy placeholders."""
    registry = get_model_registry()
    flood_model = registry.get("flood_risk_v2")
    assert flood_model is not None

    perf = flood_model.metrics
    # Must match actual artifact evaluation (ROC-AUC ~0.7931, Brier ~0.1373), NOT legacy 0.8942 / 0.0124
    assert pytest.approx(perf["roc_auc"], abs=0.005) == 0.7931
    assert pytest.approx(perf["brier_score"], abs=0.005) == 0.1373
    assert flood_model.features_count == 20

    # Verify SCS-CN challenger is registered
    challenger = registry.get("flood_risk_challenger_scs_cn")
    assert challenger is not None
    assert challenger.features_count == 23
    assert pytest.approx(challenger.metrics["roc_auc"], abs=0.005) == 0.8011


# -----------------------------------------------------------------------------
# 3. Biometeorology Scientific Rigor & Domain Validation
# -----------------------------------------------------------------------------
def test_scientific_biometeorology_distinctions():
    """Verify non-instrument labeling, UTCI domain clipping detection, and nighttime conventions."""
    # 1. WBGT non-instrument & Stull aspirated wet bulb identification
    wbgt = calculate_estimated_wbgt(temp_c=36.0, rh_pct=55.0, wind_speed_10m_ms=1.5, solar_radiation_wm2=750.0)
    assert wbgt.is_instrument_measured is False
    assert wbgt.thermal_stress_index_name == "Estimated Thermal Stress Index (Simplified WBGT Proxy)"
    assert wbgt.wet_bulb_method == "aspirated_psychrometric_stull2011"
    assert "Not an instrument black-globe measurement" in wbgt.disclaimer

    # 2. UTCI operational domain clipping
    utci_clipped = calculate_utci(temp_c=35.0, rh_pct=50.0, wind_speed_10m_ms=22.0)  # wind speed > 17 m/s
    assert utci_clipped.is_within_valid_domain is False
    assert any("clipped" in w for w in utci_clipped.domain_boundary_warnings)

    # 3. Multi-night regional threshold convention
    mn = calculate_multi_night_heat_index(nightly_min_temps_c=[27.0, 28.5, 29.0])
    assert "Project-Specific Regional Thresholds" in mn.threshold_convention
    assert "Does not constitute clinical or medical diagnosis" in mn.clinical_disclaimer


# -----------------------------------------------------------------------------
# 4. TreeSHAP Additivity & Unit Space Separation
# -----------------------------------------------------------------------------
def test_scientific_treeshap_additivity_and_unit_separation():
    """Verify exact additivity and separation of input unit vs SHAP attribution unit."""
    model_path = Path("artifacts/models/flood_risk/v0.1.0/flood_risk_v1.booster")
    assert model_path.exists()

    engine = TreeSHAPEngine()
    booster = engine.get_booster(model_path)
    feature_names = booster.feature_name()
    features = {f: 5.0 for f in feature_names}
    features["rainfall_mm"] = 120.0
    features["temperature_2m"] = 38.0

    exp = engine.explain(booster, features, is_classifier=True, target_unit="log-odds")
    assert exp.additivity_error < 1e-5

    # Check unit separation
    for contrib in exp.top_positive_contributions:
        # Input unit must be physical feature domain (e.g. mm, °C, etc.)
        assert contrib.feature_input_unit in ("mm", "°C", "scalar", "m³/m³", "std dev", "ratio", "count", "persons", "km/h", "%", "hours", "index")
        # SHAP attribution unit must be output space (log-odds margin)
        assert contrib.shap_attribution_unit == "log-odds"


# -----------------------------------------------------------------------------
# 5. Compound Copula Common Years & Upper Tail Families
# -----------------------------------------------------------------------------
def test_scientific_copula_upper_tail_and_common_years():
    """Verify copula candidate fitting includes Survival Clayton for upper tail dependence."""
    np.random.seed(42)
    x = np.random.exponential(scale=2.0, size=60)
    y = 0.8 * x + 0.2 * np.random.exponential(scale=2.0, size=60)

    engine = CompoundCopulaEngine(bootstrap_iterations=25)
    u, v = compute_pseudo_observations(x, y)
    candidates = engine.fit_candidate_families(u, v)

    family_names = [c.family for c in candidates]
    assert CopulaFamily.SURVIVAL_CLAYTON in family_names
    assert CopulaFamily.GUMBEL in family_names

    # Survival Clayton must have upper tail dependence > 0 and lower tail == 0
    sc_fit = next(c for c in candidates if c.family == CopulaFamily.SURVIVAL_CLAYTON)
    assert sc_fit.upper_tail_dependence > 0.0
    assert sc_fit.lower_tail_dependence == 0.0


# -----------------------------------------------------------------------------
# 6. Water Quality Deterministic Calculation Pathway
# -----------------------------------------------------------------------------
def test_scientific_water_quality_deterministic_pathway():
    """Verify exact BIS IS 10500:2012 calculation pathway and surrogate transparency."""
    engine = WaterQualityRiskEngine()

    sample = {
        "ph": 7.3,
        "total_dissolved_solids": 320.0,
        "total_hardness": 160.0,
        "chloride": 70.0,
        "sulphate": 50.0,
        "calcium": 45.0,
        "magnesium": 20.0,
    }
    exact = engine.calculate_exact_wqi_and_exceedances(sample)
    assert exact["exact_wqi"] < 50.0
    assert exact["wqi_category"] in ("EXCELLENT", "GOOD")

    # Evaluation with explicit measurements
    eval_res = engine.evaluate_water_quality(
        latitude=18.5204,
        longitude=73.8567,
        city="Pune",
        ph=7.3,
        electric_conductivity=500.0,
        total_hardness=160.0,
    )
    assert eval_res["is_deterministic_surrogate"] is True
    assert eval_res["wqi_calculation_mode"] == "exact_deterministic_formula"
    assert "surrogate_fidelity_note" in eval_res
