"""Unit tests for Sub-Daily Rainfall Disaggregation and Pluvial Flash-Flood Screening:
- Huff temporal distribution curves (Q1, Q2, Q3, Q4)
- Daily volume mass-balance conservation (sum(h_i) == P_daily)
- Method of Fragments
- Pluvial flash-flood screening index and provenance tagging
"""

import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import pytest
import numpy as np

from src.climacore.hydrology.subdaily_rainfall import (
    DataProvenance,
    disaggregate_daily_huff,
    disaggregate_method_of_fragments,
    screen_pluvial_flash_flood,
    HUFF_CURVES,
)


def test_huff_disaggregation_mass_conservation():
    """Verifies that all 4 Huff quartiles strictly conserve daily precipitation volume."""
    daily_depths = [15.0, 75.5, 180.2, 350.0]

    for p in daily_depths:
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            res = disaggregate_daily_huff(daily_rainfall_mm=p, quartile=q, n_hours=24)
            hourly = np.array(res["hourly_rainfall_mm"])

            # 1. Non-negativity
            assert np.all(hourly >= 0.0)
            # 2. Daily volume exact closure
            assert np.isclose(np.sum(hourly), p, atol=1e-5)
            assert res["volume_closure_error_mm"] < 1e-5
            # 3. Provenance tag
            assert res["provenance"] == DataProvenance.SCENARIO.value


def test_huff_quartile_peak_timings():
    """Verifies that peak intensity timings align with Huff quartile definitions."""
    daily_p = 100.0

    res_q1 = disaggregate_daily_huff(daily_p, quartile="Q1", n_hours=24)
    res_q4 = disaggregate_daily_huff(daily_p, quartile="Q4", n_hours=24)

    # Q1 peak occurs in the first 6 hours (first quartile)
    assert res_q1["peak_hour_index"] <= 6
    # Q4 peak occurs in the last 8 hours
    assert res_q4["peak_hour_index"] >= 16


def test_method_of_fragments():
    """Tests Method of Fragments disaggregation and custom weights."""
    daily_p = 120.0
    # Custom 24-hour weights
    weights = np.linspace(1.0, 24.0, 24)

    res = disaggregate_method_of_fragments(daily_p, fragment_weights=weights)
    hourly = np.array(res["hourly_rainfall_mm"])

    assert len(hourly) == 24
    assert np.all(hourly >= 0.0)
    assert np.isclose(np.sum(hourly), daily_p, atol=1e-5)
    # Monotonically increasing weights -> peak at last hour
    assert res["peak_hour_index"] == 23


def test_zero_rainfall_and_edge_cases():
    """Tests zero rainfall edge case."""
    res = disaggregate_daily_huff(0.0, quartile="Q1")
    assert res["daily_total_mm"] == 0.0
    assert np.sum(res["hourly_rainfall_mm"]) == 0.0
    assert res["volume_closure_error_mm"] == 0.0

    with pytest.raises(ValueError):
        disaggregate_daily_huff(50.0, quartile="INVALID_Q")


def test_screen_pluvial_flash_flood_hazard():
    """Tests urban pluvial flash-flood screening for pilot city."""
    # Scenario A: Mild shower
    mild_hourly = [1.0, 2.0, 1.5, 0.5] + [0.0] * 20
    res_mild = screen_pluvial_flash_flood(mild_hourly, soil_moisture_index=0.2, city_name="Mumbai")
    assert res_mild["warning_category"] in ("LOW", "MODERATE")
    assert res_mild["provenance"] == DataProvenance.CALCULATED.value

    # Scenario B: Catastrophic cloudburst coupled with spring tidal lock
    catastrophic_hourly = [5.0, 15.0, 75.0, 40.0, 20.0] + [2.0] * 19
    res_cata = screen_pluvial_flash_flood(
        catastrophic_hourly,
        soil_moisture_index=0.9,
        impervious_fraction=0.8,
        is_tidal_spring_high=True,
        city_name="Mumbai",
    )
    assert res_cata["warning_category"] == "EMERGENCY"
    assert res_cata["hazard_score"] >= 75.0
    assert res_cata["coastal_tidal_lock_active"] is True
    assert "conduit" in res_cata["engineering_boundary_limitations"].lower()
