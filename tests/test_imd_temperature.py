"""Unit tests for IMD 74-Year Historical Temperature Grids (1951-2024).

Verifies:
1. Processed Parquet table schema, row counts, and physical thermal bounds.
2. Provenance manifest checksum integrity.
3. Climatological normals and heatwave / coldwave indicator consistency.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from climacore.common.provenance import compute_sha256
from climacore.config.loader import get_project_root


@pytest.fixture
def root() -> Path:
    return get_project_root()


def test_imd_temperature_parquet_integrity(root: Path):
    """Test 74-year temperature parquet dataset and physical bounds."""
    parquet_path = root / "data" / "processed" / "imd_temperature_74y_50_cities.parquet"
    assert parquet_path.exists(), f"Temperature parquet missing at {parquet_path}"

    df = pd.read_parquet(parquet_path)
    assert len(df) >= 1_300_000, f"Expected >= 1.3M rows, got {len(df):,}"
    assert df["city"].nunique() == 50

    expected_cols = [
        "city", "date", "year", "day_of_year",
        "imd_tmax_c", "imd_tmin_c", "imd_dtr_c",
        "normal_tmax", "normal_tmin",
        "tmax_departure_c", "tmin_departure_c",
        "is_heatwave_day", "is_severe_heatwave_day",
        "is_coldwave_day", "is_tropical_night",
        "tmax_74y_percentile"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Physical thermal bounds on Indian subcontinent surface air temperature
    valid_tmax = df["imd_tmax_c"].dropna()
    valid_tmin = df["imd_tmin_c"].dropna()
    assert (valid_tmax >= -5.0).all() and (valid_tmax <= 55.0).all()
    assert (valid_tmin >= -20.0).all() and (valid_tmin <= 45.0).all()

    # Percentiles must be bounded in [0, 100]
    valid_pct = df["tmax_74y_percentile"].dropna()
    assert (valid_pct >= 0.0).all() and (valid_pct <= 100.0).all()


def test_imd_temperature_manifest_integrity(root: Path):
    """Test manifest matching SHA256 and metadata."""
    parquet_path = root / "data" / "processed" / "imd_temperature_74y_50_cities.parquet"
    manifest_path = root / "data" / "manifests" / "imd_temperature_manifest.json"

    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["source_id"] == "imd_temperature_74y_50_cities"
    assert manifest["cities_count"] == 50
    assert manifest["total_raw_binary_files"] == 148
    assert manifest["sha256"] == compute_sha256(parquet_path)


def test_imd_heatwave_coldwave_logic(root: Path):
    """Verify official IMD heatwave and coldwave criteria flags."""
    parquet_path = root / "data" / "processed" / "imd_temperature_74y_50_cities.parquet"
    df = pd.read_parquet(parquet_path)

    # All heatwave days must have Tmax >= 40°C
    hw_days = df[df["is_heatwave_day"] == 1]
    assert len(hw_days) > 1000
    assert (hw_days["imd_tmax_c"] >= 40.0).all()

    # Severe heatwaves must be subset of heatwaves
    severe_hw = df[df["is_severe_heatwave_day"] == 1]
    assert len(severe_hw) > 100
    assert (severe_hw["is_heatwave_day"] == 1).all()

    # Coldwaves must have Tmin <= 10°C
    cw_days = df[df["is_coldwave_day"] == 1]
    assert len(cw_days) > 1000
    assert (cw_days["imd_tmin_c"] <= 10.0).all()
