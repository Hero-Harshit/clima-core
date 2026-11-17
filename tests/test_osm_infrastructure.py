"""
Unit tests for OpenStreetMap (OSM) Critical Infrastructure Exposure Dataset.
Validates multi-buffer extraction (5km, 15km), hospital/school/bridge counts,
population ratios, SHA-256 manifests, and persona integration.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from climacore.config.loader import get_project_root
from climacore.personas.models import PersonaInput
from climacore.personas.engine import evaluate_persona
from climacore.ingestion.engine import LiveForecastEngine


@pytest.fixture
def root_dir() -> Path:
    return get_project_root()


def test_osm_processed_files_exist_and_manifest_valid(root_dir: Path):
    """Ensure parquet, json, and SHA-256 manifest files exist and are valid."""
    parquet_path = root_dir / "data" / "processed" / "infrastructure" / "city_infrastructure_exposure.parquet"
    json_path = root_dir / "data" / "processed" / "infrastructure" / "city_infrastructure_exposure.json"
    manifest_path = root_dir / "data" / "manifests" / "osm_infrastructure_manifest.json"

    assert parquet_path.exists(), f"Missing {parquet_path}"
    assert json_path.exists(), f"Missing {json_path}"
    assert manifest_path.exists(), f"Missing {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["source_id"] == "openstreetmap_geofabrik_india_infrastructure"
    assert manifest["cities_evaluated"] == 50
    assert manifest["total_hospitals_15km"] > 10_000
    assert manifest["total_schools_15km"] > 10_000
    assert manifest["total_bridges_15km"] > 25_000
    assert len(manifest["raw_zone_checksums"]) == 6
    assert len(manifest["sha256_parquet"]) == 64


def test_osm_city_coverage_and_monotonicity(root_dir: Path):
    """Ensure all 50 target cities have non-zero infrastructure and buffer monotonicity."""
    df = pd.read_parquet(root_dir / "data" / "processed" / "infrastructure" / "city_infrastructure_exposure.parquet")

    assert len(df) == 50
    required_cols = [
        "city", "state", "latitude", "longitude",
        "hospitals_5km", "hospitals_15km",
        "schools_5km", "schools_15km",
        "bridges_5km", "bridges_15km",
        "drainage_canals_5km", "drainage_canals_15km",
        "arterial_roads_15km", "industrial_zones_15km",
        "people_per_hospital_15km", "people_per_school_15km",
        "critical_infrastructure_density_km2"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column {col}"

    for _, row in df.iterrows():
        # Monotonicity: 5km <= 15km
        assert row["hospitals_5km"] <= row["hospitals_15km"]
        assert row["schools_5km"] <= row["schools_15km"]
        assert row["bridges_5km"] <= row["bridges_15km"]
        assert row["drainage_canals_5km"] <= row["drainage_canals_15km"]

        # Basic infrastructure sanity for all 50 cities
        assert row["hospitals_15km"] > 0, f"Zero hospitals for {row['city']}"
        assert row["schools_15km"] > 0, f"Zero schools for {row['city']}"
        assert row["bridges_15km"] > 0, f"Zero bridges for {row['city']}"
        assert row["people_per_hospital_15km"] > 0
        assert row["people_per_school_15km"] > 0


def test_osm_major_metros_infrastructure_values(root_dir: Path):
    """Verify ground-truth sanity bounds for major metropolitan centers."""
    df = pd.read_parquet(root_dir / "data" / "processed" / "infrastructure" / "city_infrastructure_exposure.parquet")
    city_map = {r["city"]: r for _, r in df.iterrows()}

    delhi = city_map["Delhi"]
    assert delhi["hospitals_15km"] >= 400
    assert delhi["schools_15km"] >= 800
    assert delhi["bridges_15km"] >= 2000

    bengaluru = city_map["Bengaluru"]
    assert bengaluru["hospitals_15km"] >= 800
    assert bengaluru["schools_15km"] >= 1500

    mumbai = city_map["Mumbai"]
    assert mumbai["hospitals_15km"] >= 600
    assert mumbai["bridges_15km"] >= 1500

    kolkata = city_map["Kolkata"]
    assert kolkata["hospitals_15km"] >= 250
    assert kolkata["bridges_15km"] >= 1000

    lucknow = city_map["Lucknow"]
    assert lucknow["hospitals_15km"] >= 200
    assert lucknow["schools_15km"] >= 150


def test_osm_persona_integration():
    """Verify that Persona adapters consume OSM infrastructure metrics correctly."""
    p_input = PersonaInput(
        city="Delhi",
        date="2026-06-15",
        temp_max=44.0,
        temp_min=29.0,
        precipitation_mm=45.0,
        rain_probability=0.85,
        pm2_5=210.0,
        imd_heatwave_alert=2,
        historical_percentile=98.5,
        population_15km=14_977_000,
        dense_cluster_pop_15km=14_492_000,
        urban_exposure_tier="EXTREME_METROPOLITAN_DENSITY",
        hospitals_15km=506,
        clinics_15km=450,
        total_healthcare_15km=956,
        schools_15km=1079,
        bridges_15km=2515,
        drainage_canals_15km=770,
        arterial_roads_15km=1500,
        people_per_hospital_15km=29599,
        people_per_school_15km=13881,
        critical_infrastructure_density_km2=6.43,
        drainage_arterial_ratio=0.513,
    )

    # 1. City Official
    res_city = evaluate_persona("city_official", p_input)
    assert "critical_infrastructure_footprint" in res_city.key_metrics
    assert "506 hospitals" in res_city.key_metrics["critical_infrastructure_footprint"]
    assert "drainage_discharge_resilience" in res_city.key_metrics
    assert any("bridges/flyovers" in adv for adv in res_city.advisories)

    # 2. Healthcare
    res_health = evaluate_persona("healthcare", p_input)
    assert "hospitals_in_exposure_zone" in res_health.key_metrics
    assert "506 hospitals" in res_health.key_metrics["hospitals_in_exposure_zone"]
    assert "average_surge_load_per_hospital" in res_health.key_metrics

    # 3. Student
    res_student = evaluate_persona("student", p_input)
    assert "schools_in_district_zone" in res_student.key_metrics
    assert "1,079 schools" in res_student.key_metrics["schools_in_district_zone"]

    # 4. Water
    res_water = evaluate_persona("water", p_input)
    assert "mapped_drainage_channels" in res_water.key_metrics
    assert "770 canals" in res_water.key_metrics["mapped_drainage_channels"]


def test_live_engine_infrastructure_metadata_injection():
    """Verify that LiveForecastEngine loads OSM infrastructure profiles correctly."""
    engine = LiveForecastEngine()
    engine._load_auxiliary_metadata()

    assert engine._infrastructure_profiles is not None
    assert len(engine._infrastructure_profiles) == 50
    assert "delhi" in engine._infrastructure_profiles
    assert "mumbai" in engine._infrastructure_profiles
    assert "lucknow" in engine._infrastructure_profiles

    delhi_infra = engine._infrastructure_profiles["delhi"]
    assert delhi_infra["hospitals_15km"] > 400
    assert delhi_infra["bridges_15km"] > 2000
