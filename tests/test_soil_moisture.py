"""Unit tests for ERA5-Land Soil Moisture & Agricultural Drought Intelligence.

Verifies:
1. Hydrological Parquet dataset and boundary GeoJSON integrity.
2. Root-Zone Soil Moisture LightGBM Regressor artifact, reload parity, and physical bounds.
3. Agricultural Flash Drought LightGBM Classifier artifact, probabilities, and ROC-AUC.
4. NWIC/India-WRIS groundwater and reservoir telemetry assets.
"""

import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from climacore.common.provenance import compute_sha256
from climacore.config.loader import get_project_root


@pytest.fixture
def root() -> Path:
    return get_project_root()


def test_era5_soil_moisture_dataset_integrity(root: Path):
    """Test ERA5-Land soil moisture parquet dataset and manifest."""
    parquet_path = root / "data" / "processed" / "era5_soil_moisture_50_cities.parquet"
    manifest_path = root / "data" / "manifests" / "era5_soil_moisture_manifest.json"

    assert parquet_path.exists(), f"Soil moisture parquet missing at {parquet_path}"
    assert manifest_path.exists(), f"Manifest missing at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["source_id"] == "era5_land_soil_moisture_50_cities"
    assert manifest["cities_count"] >= 14
    assert manifest["sha256"] == compute_sha256(parquet_path)

    df = pd.read_parquet(parquet_path)
    assert len(df) >= 200000, f"Expected >=200k rows, got {len(df):,}"
    expected_cols = [
        "city", "date", "soil_moisture_0_to_7cm_mean",
        "soil_moisture_7_to_28cm_mean", "soil_temperature_0_to_7cm_mean",
        "topsoil_moisture_anomaly", "rootzone_moisture_anomaly",
        "soil_moisture_deficit_index", "is_flash_drought_day"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Physical volumetric water content check (0 to 1 m3/m3)
    assert df["soil_moisture_0_to_7cm_mean"].between(0.0, 1.0).all()
    assert df["soil_moisture_7_to_28cm_mean"].between(0.0, 1.0).all()


def test_rootzone_regressor_model(root: Path):
    """Test Root-Zone Soil Moisture regressor artifact and inference sanity."""
    model_path = root / "artifacts" / "models" / "soil_moisture_drought" / "v0.1.0" / "rootzone_model.txt"
    report_path = root / "reports" / "evaluation" / "soil_moisture_drought_evaluation_report.json"

    assert model_path.exists(), f"Rootzone model missing at {model_path}"
    assert report_path.exists(), f"Evaluation report missing at {report_path}"

    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["model_name"] == "climacore_soil_moisture_drought"
    m = rep["metrics"]["rootzone_regressor"]
    assert m["r2"] > 0.90, f"Expected R2 > 0.90, got {m['r2']}"
    assert m["mae"] < m["persistence_mae"], "Model must outperform persistence baseline"
    assert m["reload_parity_diff"] == 0.0

    booster = lgb.Booster(model_file=str(model_path))
    num_features = booster.num_feature()
    assert num_features > 0

    dummy_input = np.ones((5, num_features)) * 0.25
    preds = booster.predict(dummy_input)
    assert preds.shape == (5,)
    assert not np.isnan(preds).any()
    # Soil moisture is positive volumetric fraction
    assert (preds >= 0.0).all()


def test_flash_drought_classifier_model(root: Path):
    """Test Agricultural Flash Drought classifier artifact and probabilities."""
    model_path = root / "artifacts" / "models" / "soil_moisture_drought" / "v0.1.0" / "flash_drought_model.txt"
    report_path = root / "reports" / "evaluation" / "soil_moisture_drought_evaluation_report.json"

    assert model_path.exists(), f"Flash drought model missing at {model_path}"

    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    m = rep["metrics"]["flash_drought_classifier"]
    assert m["roc_auc"] > 0.90, f"Expected ROC-AUC > 0.90, got {m['roc_auc']}"
    assert m["accuracy"] > 0.90, f"Expected Accuracy > 90%, got {m['accuracy']}"
    assert m["reload_parity_diff"] == 0.0

    booster = lgb.Booster(model_file=str(model_path))
    num_features = booster.num_feature()
    assert num_features > 0

    dummy_input = np.ones((5, num_features)) * 0.10
    probs = booster.predict(dummy_input)
    assert probs.shape == (5,)
    assert not np.isnan(probs).any()
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_geospatial_boundaries_and_water_manifest(root: Path):
    """Test district boundaries and NWIC water datasets."""
    district_geojson = root / "data" / "processed" / "boundaries" / "india_districts.geojson"
    gw_parquet = root / "data" / "processed" / "cgwb_groundwater_levels.parquet"
    res_parquet = root / "data" / "processed" / "cwc_reservoir_storage.parquet"

    assert district_geojson.exists()
    assert gw_parquet.exists()
    assert res_parquet.exists()

    with open(district_geojson, "r", encoding="utf-8") as f:
        geo = json.load(f)
    assert geo["type"] == "FeatureCollection"
    assert len(geo["features"]) > 500

    gw_df = pd.read_parquet(gw_parquet)
    assert len(gw_df) > 400000

    res_df = pd.read_parquet(res_parquet)
    assert len(res_df) > 1000000
