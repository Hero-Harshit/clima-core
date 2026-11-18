"""
Unit tests for India Flood Inventory (IFI) disaster data ingestion and LightGBM Flood Risk Classifier.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import pytest

from climacore.config.loader import get_project_root
from climacore.data.sources.ifi_floods import (
    CITY_IFI_DISTRICTS_MAP,
    compute_api_series,
)
from climacore.models.flood_risk.train import FLOOD_FEATURES


def test_city_ifi_districts_map_complete():
    """Verify all 50 target cities have defined administrative aliases in IFI map."""
    assert len(CITY_IFI_DISTRICTS_MAP) == 50
    for city, aliases in CITY_IFI_DISTRICTS_MAP.items():
        assert len(aliases) >= 1
        assert all(isinstance(a, str) and len(a) > 0 for a in aliases)


def test_compute_api_series():
    """Verify Antecedent Precipitation Index exponential decay computation."""
    rain = np.array([0.0, 10.0, 0.0, 0.0, 20.0], dtype=np.float32)
    api = compute_api_series(rain, decay=0.85)
    assert len(api) == 5
    assert api[0] == 0.0
    assert api[1] == 10.0
    assert np.isclose(api[2], 8.5, atol=1e-3)
    assert np.isclose(api[3], 7.225, atol=1e-3)
    assert np.isclose(api[4], 20.0 + 0.85 * 7.225, atol=1e-3)


def test_aligned_flood_dataset_exists_and_valid():
    """Verify aligned flood parquet dataset and manifest."""
    root = get_project_root()
    parquet_path = root / "data" / "processed" / "floods" / "city_flood_history_aligned.parquet"
    manifest_path = root / "data" / "manifests" / "ifi_flood_manifest.json"

    assert parquet_path.exists(), f"Aligned flood dataset not found at {parquet_path}"
    assert manifest_path.exists(), f"IFI manifest not found at {manifest_path}"

    df = pd.read_parquet(parquet_path)
    assert len(df) >= 700_000
    assert df["city"].nunique() == 50
    assert df["flood_disaster_event"].sum() >= 5_000

    for feat in FLOOD_FEATURES:
        assert feat in df.columns, f"Missing required feature: {feat}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["total_cities"] == 50
    assert manifest["positive_flood_days"] >= 5_000


def test_flood_risk_model_artifacts_and_metrics():
    """Verify trained LightGBM flood booster and evaluation metrics."""
    root = get_project_root()
    booster_path = root / "artifacts" / "models" / "flood_risk" / "v0.1.0" / "flood_risk_v1.booster"
    report_path = root / "reports" / "evaluation" / "flood_risk_evaluation_report.json"

    assert booster_path.exists(), f"Trained flood booster not found at {booster_path}"
    assert report_path.exists(), f"Flood evaluation report not found at {report_path}"

    booster = lgb.Booster(model_file=str(booster_path))
    assert booster.num_feature() == len(FLOOD_FEATURES)

    # Test inference on dummy input
    dummy_input = np.ones((2, len(FLOOD_FEATURES)), dtype=np.float32) * 5.0
    preds = booster.predict(dummy_input)
    assert len(preds) == 2
    assert all(0.0 <= p <= 1.0 for p in preds)

    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    metrics = rep["metrics"]
    assert metrics["test_roc_auc"] >= 0.75
    assert metrics["test_recall"] >= 0.60
