"""
Unit tests for WorldPop 2024 100m Population Exposure Grid Integration.
Validates multi-scale concentric buffer extraction (5km, 15km, 30km), informal settlement
density thresholds (>=7,500/km2), metadata manifests, and persona integration.
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


def test_worldpop_processed_files_exist_and_manifest_valid(root_dir: Path):
    """Ensure parquet, json, and sha256 manifest files are generated and valid."""
    parquet_path = root_dir / "data" / "processed" / "population" / "city_population_exposure.parquet"
    json_path = root_dir / "data" / "processed" / "population" / "city_population_exposure.json"
    manifest_path = root_dir / "data" / "manifests" / "worldpop_manifest.json"

    assert parquet_path.exists(), f"Missing {parquet_path}"
    assert json_path.exists(), f"Missing {json_path}"
    assert manifest_path.exists(), f"Missing {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["source_id"] == "worldpop_india_2024_100m_constrained"
    assert manifest["cities_evaluated"] == 50
    assert "EPSG:4326" in manifest["crs"]
    assert len(manifest["sha256_raw"]) == 64
    assert len(manifest["sha256_parquet"]) == 64


def test_worldpop_city_coverage_and_monotonicity(root_dir: Path):
    """Ensure all 50 target cities are present with concentric radial monotonicity."""
    df = pd.read_parquet(root_dir / "data" / "processed" / "population" / "city_population_exposure.parquet")

    assert len(df) == 50
    required_cols = [
        "city", "state", "latitude", "longitude",
        "population_5km", "density_5km_per_km2",
        "population_15km", "density_15km_per_km2",
        "dense_cluster_pop_15km", "peak_cell_population_100m",
        "population_30km", "density_30km_per_km2",
        "urban_exposure_tier"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column {col}"

    # Verify concentric monotonicity: 5km <= 15km <= 30km
    for _, row in df.iterrows():
        p5 = row["population_5km"]
        p15 = row["population_15km"]
        p30 = row["population_30km"]
        dense = row["dense_cluster_pop_15km"]

        assert 0 < p5 <= p15 <= p30, f"Monotonicity violated for {row['city']}: {p5} <= {p15} <= {p30}"
        assert 0 <= dense <= p15, f"Dense cluster exceeds 15km population for {row['city']}: {dense} > {p15}"
        assert row["density_15km_per_km2"] > 0.0
        assert row["urban_exposure_tier"] in [
            "EXTREME_METROPOLITAN_DENSITY",
            "HIGH_URBAN_EXPOSURE",
            "MODERATE_TIER2_EXPOSURE",
            "TIER3_EMERGING_URBAN"
        ]


def test_worldpop_major_metros_metrics(root_dir: Path):
    """Verify ground-truth sanity bounds for major metropolitan centers."""
    df = pd.read_parquet(root_dir / "data" / "processed" / "population" / "city_population_exposure.parquet")
    city_map = {r["city"]: r for _, r in df.iterrows()}

    delhi = city_map["Delhi"]
    assert delhi["population_15km"] >= 10_000_000
    assert delhi["dense_cluster_pop_15km"] >= 5_000_000
    assert delhi["urban_exposure_tier"] == "EXTREME_METROPOLITAN_DENSITY"

    kolkata = city_map["Kolkata"]
    assert kolkata["population_15km"] >= 5_000_000
    assert kolkata["urban_exposure_tier"] in ["HIGH_URBAN_EXPOSURE", "EXTREME_METROPOLITAN_DENSITY"]

    mumbai = city_map["Mumbai"]
    assert mumbai["population_15km"] >= 2_000_000
    assert mumbai["density_5km_per_km2"] >= 5_000.0

    bengaluru = city_map["Bengaluru"]
    assert bengaluru["population_15km"] >= 3_500_000


def test_population_integration_in_personas():
    """Verify that Persona adapters consume population exposure metrics correctly."""
    p_input = PersonaInput(
        city="Delhi",
        date="2026-06-15",
        temp_max=44.0,
        temp_min=29.0,
        precipitation_mm=0.0,
        pm2_5=210.0,
        imd_heatwave_alert=2,
        historical_percentile=98.5,
        population_5km=1_690_000,
        population_15km=14_977_000,
        population_30km=30_160_000,
        dense_cluster_pop_15km=14_492_000,
        urban_exposure_tier="EXTREME_METROPOLITAN_DENSITY",
    )

    # 1. City Official Persona
    res_city = evaluate_persona("city_official", p_input)
    assert "worldpop_population_15km" in res_city.key_metrics
    assert "14,977,000" in res_city.key_metrics["worldpop_population_15km"]
    assert "dense_cluster_population_15km" in res_city.key_metrics
    assert "EXTREME_METROPOLITAN_DENSITY" == res_city.key_metrics["urban_exposure_tier"]
    assert any("informal settlement clusters" in adv.lower() for adv in res_city.advisories)

    # 2. Healthcare Persona
    res_health = evaluate_persona("healthcare", p_input)
    assert "exposed_population_15km" in res_health.key_metrics
    assert "vulnerable_demographic_pool" in res_health.key_metrics
    assert "projected_emergency_triage_demand" in res_health.key_metrics
    assert any("elderly and asthmatic patients projected" in adv.lower() for adv in res_health.advisories)

    # 3. Insurance Persona
    res_ins = evaluate_persona("insurance", p_input)
    assert "portfolio_population_exposure" in res_ins.key_metrics
    assert "informal_settlement_claim_potential" in res_ins.key_metrics
    assert "urban_concentration_loss_multiplier" in res_ins.key_metrics
    assert "x" in res_ins.key_metrics["urban_concentration_loss_multiplier"]


def test_live_engine_population_metadata_injection():
    """Verify that LiveForecastEngine loads WorldPop population metadata seamlessly."""
    engine = LiveForecastEngine()
    engine._load_auxiliary_metadata()

    assert engine._population_profiles is not None
    assert len(engine._population_profiles) == 50
    assert "delhi" in engine._population_profiles
    assert "mumbai" in engine._population_profiles

    delhi_pop = engine._population_profiles["delhi"]
    assert delhi_pop["population_15km"] > 10_000_000
    assert delhi_pop["urban_exposure_tier"] == "EXTREME_METROPOLITAN_DENSITY"
