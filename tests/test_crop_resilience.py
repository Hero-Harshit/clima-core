"""
Unit tests for AgriSense India District Crop & Soil Resilience Ingestion and LightGBM Suitability Model.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import pytest

from climacore.config.loader import get_project_root
from climacore.models.crop_resilience.train import CROP_SOIL_FEATURES, TARGET_CROPS
from climacore.personas.models import PersonaInput
from climacore.personas.farmer import evaluate_farmer_persona


def test_city_district_crop_profiles_exist_and_complete():
    """Verify city crop resilience parquet and JSON profiles cover all 50 cities."""
    root = get_project_root()
    parquet_path = root / "data" / "processed" / "agriculture" / "city_district_crop_profiles.parquet"
    json_path = root / "data" / "processed" / "agriculture" / "city_district_crop_profiles.json"
    manifest_path = root / "data" / "manifests" / "crop_dataset_manifest.json"

    assert parquet_path.exists(), f"Crop profiles parquet missing at {parquet_path}"
    assert json_path.exists(), f"Crop profiles JSON missing at {json_path}"
    assert manifest_path.exists(), f"Crop manifest missing at {manifest_path}"

    df = pd.read_parquet(parquet_path)
    assert len(df) == 50
    assert df["city"].nunique() == 50

    # Verify physical bounds of soil characteristics
    assert df["soil_ph"].between(4.0, 10.0).all(), "Soil pH out of realistic agronomic range"
    assert df["soil_organic_carbon_g_kg"].between(1.0, 120.0).all()
    assert df["soil_fertility_index"].between(10.0, 100.0).all()
    assert df["soil_water_holding_capacity"].between(5.0, 60.0).all()

    # Verify dominant crop values
    assert df["dominant_kharif_crop"].notna().all()
    assert df["dominant_rabi_crop"].notna().all()

    with open(json_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    assert len(profiles) == 50


def test_crop_suitability_model_artifacts_and_metrics():
    """Verify trained LightGBM crop suitability booster and evaluation metrics."""
    root = get_project_root()
    booster_path = root / "artifacts" / "models" / "crop_resilience" / "v0.1.0" / "crop_suitability_v1.booster"
    report_path = root / "reports" / "evaluation" / "crop_resilience_evaluation_report.json"

    assert booster_path.exists(), f"Trained crop booster not found at {booster_path}"
    assert report_path.exists(), f"Crop evaluation report not found at {report_path}"

    booster = lgb.Booster(model_file=str(booster_path))
    assert booster.num_feature() == len(CROP_SOIL_FEATURES)

    # Test inference: output probabilities over 10 crop classes
    dummy_input = np.ones((1, len(CROP_SOIL_FEATURES)), dtype=np.float32) * 15.0
    probs = booster.predict(dummy_input)
    assert probs.shape == (1, len(TARGET_CROPS))
    assert np.isclose(np.sum(probs), 1.0, atol=1e-3)

    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    metrics = rep["metrics"]
    assert metrics["top1_test_accuracy"] >= 0.75
    assert metrics["top3_test_accuracy"] >= 0.90
    assert metrics["weighted_f1"] >= 0.75


def test_farmer_persona_crop_intelligence_integration():
    """Verify Farmer persona incorporates soil and crop-specific advisories."""
    inp = PersonaInput(
        city="Ludhiana",
        date="2026-03-15",
        temp_max=34.5,
        temp_min=18.0,
        precipitation_mm=0.0,
        rain_probability=0.05,
        pm2_5=45.0,
        agricultural_district="Ludhiana",
        dominant_kharif_crop="Rice",
        dominant_rabi_crop="Wheat",
        dominant_crop_overall="Wheat",
        soil_ph=7.8,
        soil_fertility_index=68.5,
        soil_water_holding_capacity=22.4,
    )

    res = evaluate_farmer_persona(inp)
    assert res.persona_id == "farmer_kisan"
    assert "agricultural_hinterland" in res.key_metrics
    assert "Wheat" in res.key_metrics["agricultural_hinterland"]
    assert "dominant_cropping_pattern" in res.key_metrics
    assert "soil_fertility_and_whc" in res.key_metrics

    # Check terminal heat stress alert on wheat when temp_max >= 32°C
    advisories_str = " ".join(res.advisories)
    assert "AgriSense Wheat Alert" in advisories_str or "heat" in advisories_str.lower()
