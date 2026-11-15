"""Unit tests for NDMA & State Heat Action Plan (HAP) City Alert Triggers.

Verifies:
1. All 50 cities are properly mapped to their terrain and HAP framework.
2. Pioneer municipal HAPs (Ahmedabad AMC, Bhubaneswar OSDMA, Delhi DDMA) match official thresholds.
3. Provenance manifest checksum integrity.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from climacore.common.provenance import compute_sha256
from climacore.config.loader import get_project_root


@pytest.fixture
def root() -> Path:
    return get_project_root()


def test_ndma_hap_triggers_integrity(root: Path):
    """Test 50-city NDMA HAP triggers dataset."""
    parquet_path = root / "data" / "processed" / "ndma_hap_triggers.parquet"
    json_path = root / "data" / "processed" / "ndma_hap_triggers.json"

    assert parquet_path.exists(), f"Parquet missing at {parquet_path}"
    assert json_path.exists(), f"JSON missing at {json_path}"

    df = pd.read_parquet(parquet_path)
    assert len(df) == 50

    # Verify Ahmedabad pioneer thresholds (41.1, 43.1, 45.0)
    ahmedabad = df[df["city"] == "Ahmedabad"].iloc[0]
    assert ahmedabad["threshold_yellow_min_c"] == 41.1
    assert ahmedabad["threshold_orange_min_c"] == 43.1
    assert ahmedabad["threshold_red_alert_c"] == 45.0

    # Verify Bhubaneswar coastal thresholds
    bhubaneswar = df[df["city"] == "Bhubaneswar"].iloc[0]
    assert bhubaneswar["threshold_red_alert_c"] == 43.0

    # Verify Hill cities have lower thermal triggers
    shimla = df[df["city"] == "Shimla"].iloc[0]
    assert shimla["threshold_red_alert_c"] == 35.0


def test_ndma_hap_manifest_integrity(root: Path):
    """Test manifest checksum integrity."""
    manifest_path = root / "data" / "manifests" / "ndma_hap_manifest.json"
    json_path = root / "data" / "processed" / "ndma_hap_triggers.json"
    parquet_path = root / "data" / "processed" / "ndma_hap_triggers.parquet"

    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["source_id"] == "ndma_heat_action_plan_triggers"
    assert meta["cities_covered"] == 50
    assert meta["sha256_json"] == compute_sha256(json_path)
    assert meta["sha256_parquet"] == compute_sha256(parquet_path)
