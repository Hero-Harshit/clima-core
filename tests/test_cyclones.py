"""Unit tests for IMD North Indian Ocean Cyclone Best Tracks & City Risk Engine.

Verifies:
1. Master track points parquet schema, row count, and coordinate physical bounds.
2. 50-city cyclone risk profiles (exposure buffers, hazard tiers, known coastal risks).
3. Provenance manifest checksum integrity.
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


def test_cyclone_master_tracks_integrity(root: Path):
    """Test 1982-2026 master cyclone trajectory points."""
    tracks_path = root / "data" / "processed" / "cyclones" / "north_indian_ocean_cyclone_tracks_1982_2026.parquet"
    assert tracks_path.exists(), f"Tracks parquet missing at {tracks_path}"

    df = pd.read_parquet(tracks_path)
    assert len(df) >= 3_000, f"Expected at least 3,000 track points, got {len(df):,}"
    assert df["year"].min() <= 1985
    assert df["year"].max() >= 2024

    expected_cols = [
        "system_id", "year", "basin", "name",
        "timestamp_utc", "date", "latitude", "longitude",
        "central_pressure_hpa", "max_wind_kt", "max_wind_kmh",
        "pressure_drop_hpa", "grade"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Lat/Lon bounds for North Indian Ocean (Arabian Sea + Bay of Bengal + landfall areas)
    valid_lats = df["latitude"].dropna()
    valid_lons = df["longitude"].dropna()
    assert (valid_lats >= 0.0).all() and (valid_lats <= 36.0).all()
    assert (valid_lons >= 40.0).all() and (valid_lons <= 105.0).all()

    # IMD Scale grades
    valid_grades = {"LOW", "D", "DD", "CS", "SCS", "VSCS", "ESCS", "SuCS"}
    assert set(df["grade"].unique()).issubset(valid_grades)


def test_city_cyclone_risk_profiles(root: Path):
    """Test 50-city cyclone risk exposure buffers and hazard classifications."""
    risk_path = root / "data" / "processed" / "cyclones" / "city_cyclone_risk_profiles.parquet"
    assert risk_path.exists(), f"City risk parquet missing at {risk_path}"

    df = pd.read_parquet(risk_path)
    assert len(df) == 50, f"Expected 50 cities, got {len(df)}"

    expected_cols = [
        "city", "latitude", "longitude", "cyclone_hazard_tier",
        "total_storms_500km", "total_storms_250km", "direct_strikes_100km",
        "severe_cyclones_250km", "min_historic_distance_km",
        "peak_historic_wind_kmh", "closest_storm"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Coastal vs Inland hazard tiers:
    # Coastal hubs like Chennai, Visakhapatnam, Kolkata, Bhubaneswar, Surat must have higher risk
    # than inland/Himalayan cities like Srinagar, Leh, Shimla
    city_tier_map = dict(zip(df["city"], df["cyclone_hazard_tier"]))
    high_risk_coastal = ["Chennai", "Visakhapatnam", "Kolkata", "Bhubaneswar", "Puri", "Surat"]
    sheltered_inland = ["Srinagar", "Shimla", "Dehradun"]

    for c in high_risk_coastal:
        if c in city_tier_map:
            assert city_tier_map[c] in ["EXTREME_COASTAL_VULNERABILITY", "HIGH_CYCLONE_HAZARD"], (
                f"{c} should be classified as high/extreme cyclone hazard, got {city_tier_map[c]}"
            )

    for c in sheltered_inland:
        if c in city_tier_map:
            assert city_tier_map[c] in ["LOW_SHELTERED", "MODERATE_INLAND_DEPRESSION_RISK"], (
                f"{c} should be classified as low/sheltered, got {city_tier_map[c]}"
            )

    # Distances must be non-negative
    assert (df["min_historic_distance_km"] >= 0.0).all()
    # Monotonicity of buffer counts: 100km <= 250km <= 500km
    assert (df["direct_strikes_100km"] <= df["total_storms_250km"]).all()
    assert (df["total_storms_250km"] <= df["total_storms_500km"]).all()


def test_cyclones_manifest_integrity(root: Path):
    """Test manifest matching SHA256 and metadata."""
    manifest_path = root / "data" / "manifests" / "cyclones_manifest.json"
    tracks_path = root / "data" / "processed" / "cyclones" / "north_indian_ocean_cyclone_tracks_1982_2026.parquet"
    risk_path = root / "data" / "processed" / "cyclones" / "city_cyclone_risk_profiles.parquet"

    assert manifest_path.exists(), f"Manifest missing at {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["source_id"] == "imd_cyclone_best_tracks_1982_2026"
    assert meta["cities_assessed"] == 50
    assert meta["total_cyclonic_systems"] > 0
    assert meta["total_trajectory_points"] > 0

    assert meta["sha256_tracks_parquet"] == compute_sha256(tracks_path)
    assert meta["sha256_city_risk_parquet"] == compute_sha256(risk_path)
