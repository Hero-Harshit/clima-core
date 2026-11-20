"""Unit tests for Terrain and Geospatial Exposure Intelligence:
- DEM slope, aspect, and gradient derivation
- Topographic Wetness Index (TWI) calculation and boundary clamping
- HydroRIVERS network Strahler ordering and corridor hazard decay
- Civil lifelines critical asset exposure and TIVI composite scoring
"""

import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import pytest
import numpy as np

from src.climacore.geospatial.terrain_intelligence import (
    compute_slope_and_aspect,
    compute_topographic_wetness_index,
    HydroRiverNetwork,
    AssetExposureAnalyzer,
)


def test_slope_and_aspect_tilted_plane():
    """Verifies slope and aspect on an exact mathematical tilted plane."""
    # 5x5 grid with cell_size = 10m
    # Elevation drops 1 meter per 10 meters in x direction (gradient = 0.1)
    grid = np.array([
        [10.0, 11.0, 12.0, 13.0, 14.0],
        [10.0, 11.0, 12.0, 13.0, 14.0],
        [10.0, 11.0, 12.0, 13.0, 14.0],
        [10.0, 11.0, 12.0, 13.0, 14.0],
        [10.0, 11.0, 12.0, 13.0, 14.0],
    ])

    res = compute_slope_and_aspect(grid, cell_size_m=10.0)

    # Interior cells (row 2, col 2)
    interior_slope_deg = res["slope_degrees"][2, 2]
    expected_deg = np.degrees(np.arctan(0.1))
    assert np.isclose(interior_slope_deg, expected_deg, atol=1e-2)

    interior_pct = res["slope_percent"][2, 2]
    assert np.isclose(interior_pct, 10.0, atol=1e-2)

    assert np.all(res["slope_degrees"] >= 0.0)
    assert np.all(res["slope_degrees"] <= 90.0)


def test_topographic_wetness_index_bounds():
    """Verifies TWI monotonicity and flat-terrain non-singularity."""
    slope_flat = np.array([0.0, 0.001, 0.05, 0.2, 0.5])  # radians
    a_contrib = 100.0  # m2/m

    twi = compute_topographic_wetness_index(slope_flat, specific_contributing_area_m=a_contrib)

    # 1. No NaNs or infinities even at 0.0 slope
    assert not np.any(np.isnan(twi))
    assert not np.any(np.isinf(twi))

    # 2. Flatter terrain must have higher TWI (greater waterlogging potential)
    assert twi[0] > twi[-1]

    # 3. Higher contributing area must increase TWI
    twi_high_a = compute_topographic_wetness_index(slope_flat, specific_contributing_area_m=1000.0)
    assert np.all(twi_high_a > twi)


def test_hydro_river_network_corridor_hazard():
    """Tests HydroRIVERS reach filtering and distance decay hazard score."""
    sample_reaches = [
        {"reach_id": 101, "river_name": "Mithi", "strahler_order": 3},
        {"reach_id": 102, "river_name": "Ulhas", "strahler_order": 5},
        {"reach_id": 103, "river_name": "Godavari", "strahler_order": 7},
    ]

    net = HydroRiverNetwork(sample_reaches)
    major_rivers = net.filter_by_strahler(min_order=5)
    assert len(major_rivers) == 2

    # Proximity score tests:
    # 50m from order 5 river vs 500m vs 3000m
    score_close = net.compute_corridor_hazard_score(distance_to_reach_m=50.0, strahler_order=5)
    score_far = net.compute_corridor_hazard_score(distance_to_reach_m=500.0, strahler_order=5)
    score_outside = net.compute_corridor_hazard_score(distance_to_reach_m=3000.0, strahler_order=5)

    assert 0.0 <= score_close <= 100.0
    assert score_close > score_far
    assert score_outside == 0.0


def test_asset_exposure_analyzer_tivi():
    """Tests critical asset weighting and Terrain & Infrastructure Vulnerability Index (TIVI)."""
    analyzer = AssetExposureAnalyzer()

    # Scenario 1: Low vulnerability (high slope, low TWI, far from river, low assets)
    res_low = analyzer.compute_terrain_infrastructure_vulnerability_index(
        mean_twi=5.0,
        mean_slope_deg=8.0,
        asset_counts={"hospitals": 1, "schools": 2, "bridges": 0},
        population_density_sqkm=800.0,
        distance_to_primary_river_m=5000.0,
        primary_river_strahler=3,
    )
    assert res_low["vulnerability_tier"] in ("LOW", "MODERATE")
    assert 0.0 <= res_low["terrain_infrastructure_vulnerability_index"] <= 50.0

    # Scenario 2: High vulnerability (flat coastal floodplain, high TWI, close to major river, dense lifelines)
    res_high = analyzer.compute_terrain_infrastructure_vulnerability_index(
        mean_twi=14.5,
        mean_slope_deg=0.3,
        asset_counts={"hospitals": 35, "schools": 80, "bridges": 18, "transit_stations": 24},
        population_density_sqkm=25000.0,
        distance_to_primary_river_m=120.0,
        primary_river_strahler=6,
    )
    assert res_high["vulnerability_tier"] in ("HIGH", "VERY_HIGH")
    assert res_high["terrain_infrastructure_vulnerability_index"] >= 65.0
    assert res_high["sub_indices"]["topographic_wetness_score"] > 60.0
    assert res_high["sub_indices"]["slope_stagnation_score"] > 80.0
