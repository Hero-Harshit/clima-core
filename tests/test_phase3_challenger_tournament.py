"""
Unit and Integration Test Suite for Phase 3 · Advanced ML & Hybrid Architectures.
Validates:
  1. CatBoost Crop Suitability Challenger loading, symmetric tree inference, and evaluation report.
  2. Temporal Fusion Transformer (TFT) PowerGrid Challenger loading, multi-horizon quantile inference, and report.
  3. Dynamic Model Engine Routing (Champion, Challenger, Ensemble) in AgriSenseEngine.
  4. Dynamic Model Engine Routing (Champion, Challenger, Ensemble) in PowerGridEngine.
  5. Canonical Champion vs Challenger Tournament Manifest and Guiding Rule adherence.
"""

import json
from pathlib import Path
import pytest

from climacore.config.loader import get_project_root
from climacore.models.crop_resilience.challenger_catboost import (
    get_crop_challenger_booster,
    CatBoostCropSuitabilityEngine,
)
from climacore.models.electricity_grid.challenger_tft import (
    get_tft_challenger,
    TFTGridStressEngine,
)
from climacore.models.crop_resilience.engine import AgriSenseEngine
from climacore.models.electricity_grid.engine import PowerGridEngine
from climacore.evaluation.tournament import run_full_champion_challenger_tournament


def test_catboost_crop_challenger_inference():
    """Verify CatBoost Crop Suitability Challenger loads and outputs ranked probabilities."""
    model = get_crop_challenger_booster()
    assert model is not None

    engine = CatBoostCropSuitabilityEngine()
    features = {
        "soil_ph_h2o_0_30cm": 7.2,
        "soil_total_n_g_kg_0_30cm": 1.1,
        "soil_soc_g_kg_0_30cm": 8.5,
        "soil_cec_cmol_kg_0_30cm": 15.0,
        "soil_clay_pct_0_30cm": 28.0,
        "soil_sand_pct_0_30cm": 40.0,
        "soil_silt_pct_0_30cm": 32.0,
        "soil_bulk_density_kg_dm3_0_30cm": 1.35,
        "temp_mean_c": 28.0,
        "temp_max_c": 34.0,
        "temp_min_c": 22.0,
        "humidity_pct": 65.0,
        "rainfall_mm": 5.0,
        "rain_days_proxy_mm_per_day": 5.0,
        "solar_mj_m2_day": 20.0,
        "annual_rainfall_mm": 950.0,
        "annual_temp_mean_c": 26.0,
        "annual_humidity_pct": 65.0,
        "elevation_m": 120.0,
    }

    preds = engine.predict_suitability(features)
    assert len(preds) == 5
    assert preds[0]["rank"] == 1
    assert preds[0]["probability"] > 0.0
    # Probabilities in descending order
    for i in range(len(preds) - 1):
        assert preds[i]["probability"] >= preds[i + 1]["probability"]


def test_tft_power_grid_challenger_inference():
    """Verify Temporal Fusion Transformer (TFT) loads and predicts multi-horizon quantiles."""
    model = get_tft_challenger()
    assert model is not None

    engine = TFTGridStressEngine()
    features = {
        "population_15km": 5_000_000,
        "dense_cluster_pop_15km": 750_000,
        "temp_max_c": 42.0,
        "temp_min_c": 28.0,
        "temp_mean_c": 35.0,
        "cooling_degree_days": 17.0,
        "is_tropical_night": 1.0,
        "dewpoint_mean_c": 22.0,
        "utci_mean_c": 36.0,
        "utci_max_c": 44.0,
        "solar_radiation_mj": 24.0,
        "cloud_cover_pct": 20.0,
        "day_of_week": 3,
        "is_weekend": 0.0,
        "month": 6,
    }

    quantiles = engine.predict_peak_load_quantiles(city="Nagpur", features=features)
    assert "h24" in quantiles and "h48" in quantiles and "h72" in quantiles

    # Monotonic quantile constraints: Q10 <= Q50 <= Q90
    for h in ["h24", "h48", "h72"]:
        q = quantiles[h]
        assert q["q10"] <= q["q50"] <= q["q90"]
        assert q["q50"] > 1000.0  # Realistic megawatt order


def test_agrisense_engine_mode_routing():
    """Verify AgriSenseEngine routes between champion (LightGBM), challenger (CatBoost), and ensemble."""
    engine = AgriSenseEngine()

    sample_args = {
        "city": "Lucknow",
        "doy": 150,
        "temp_max_c": 36.0,
        "temp_min_c": 24.0,
        "temp_mean_c": 30.0,
        "rainfall_mm": 5.0,
        "humidity_pct": 60.0,
    }

    res_champ = engine.evaluate_agrisense(**sample_args, engine_mode="champion")
    res_chall = engine.evaluate_agrisense(**sample_args, engine_mode="challenger")
    res_ens = engine.evaluate_agrisense(**sample_args, engine_mode="ensemble")

    assert res_champ["engine_mode"] == "champion"
    assert res_chall["engine_mode"] == "challenger"
    assert res_ens["engine_mode"] == "ensemble"

    assert len(res_champ["top_suitable_crops"]) >= 3
    assert len(res_chall["top_suitable_crops"]) >= 3
    assert len(res_ens["top_suitable_crops"]) >= 3


def test_power_grid_engine_mode_routing():
    """Verify PowerGridEngine routes between champion (LightGBM), challenger (TFT), and ensemble."""
    engine = PowerGridEngine()

    sample_args = {
        "city": "Hyderabad",
        "state": "Telangana",
        "temp_max_c": 41.0,
        "temp_min_c": 27.0,
        "population_15km": 6_500_000,
    }

    res_champ = engine.evaluate_power_grid_risk(**sample_args, engine_mode="champion")
    res_chall = engine.evaluate_power_grid_risk(**sample_args, engine_mode="challenger")
    res_ens = engine.evaluate_power_grid_risk(**sample_args, engine_mode="ensemble")

    assert res_champ["engine_mode"] == "champion"
    assert res_chall["engine_mode"] == "challenger"
    assert res_ens["engine_mode"] == "ensemble"

    assert res_champ["peak_load_mw"] > 0.0
    assert res_chall["peak_load_mw"] > 0.0
    assert res_ens["peak_load_mw"] > 0.0

    assert res_chall["tft_sequence_quantiles"] is not None
    assert res_ens["tft_sequence_quantiles"] is not None


def test_champion_challenger_tournament_manifest_integrity():
    """Verify Canonical Champion vs Challenger Tournament Manifest and Guiding Rule verdicts."""
    root = get_project_root()
    manifest_path = root / "data" / "manifests" / "champion_challenger_tournament.json"
    assert manifest_path.exists(), "Tournament manifest must be published"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "tournaments" in manifest
    assert "agrisense_crop_suitability" in manifest["tournaments"]
    assert "power_grid_peak_load" in manifest["tournaments"]

    # Verify AgriSense tournament details
    crop_t = manifest["tournaments"]["agrisense_crop_suitability"]
    assert crop_t["champion"]["model_name"] == "AgriSense LightGBM Champion"
    assert crop_t["challenger"]["model_name"] == "AgriSense CatBoost Challenger"
    assert crop_t["champion"]["top1_test_accuracy"] >= 0.82
    assert crop_t["challenger"]["top1_test_accuracy"] >= 0.79
    assert crop_t["verdict"]["status"] == "CHAMPION_RETAINS_LEAD"
    assert crop_t["operational_routing"]["primary_engine"] == "champion_lightgbm"

    # Verify PowerGrid tournament details
    grid_t = manifest["tournaments"]["power_grid_peak_load"]
    assert grid_t["champion"]["model_name"] == "PowerGrid LightGBM Champion"
    assert grid_t["challenger"]["model_name"] == "PowerGrid Temporal Fusion Transformer (TFT)"
    assert grid_t["champion"]["test_r2"] >= 0.87
    assert grid_t["challenger"]["test_r2"] >= 0.70
    assert grid_t["verdict"]["status"] == "CHAMPION_RETAINS_LEAD"
    assert grid_t["operational_routing"]["primary_engine"] == "champion_lightgbm"
