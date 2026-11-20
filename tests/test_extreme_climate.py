"""Unit tests for Extreme Climate and Non-Stationary EVT Statistics:
- Non-Stationary GEV distribution with decadal location trend
- Peaks-Over-Threshold (POT) run-length declustering and GPD return levels
- DroughtCascade HMM forward causal filtering numerical stability
"""

import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import pytest
import numpy as np

from src.climacore.analytics.extreme_climate import (
    NonStationaryGEV,
    PeaksOverThresholdGPD,
    audit_drought_cascade_forward_stability,
)


def test_non_stationary_gev_trend_detection():
    """Verifies that NonStationaryGEV correctly captures secular warming trends."""
    np.random.seed(42)
    # 50 years of annual maximum temperature (1975 to 2024)
    years = np.arange(1975, 2025)
    # Base location = 40.0 C, secular warming trend of +0.4 C per decade (0.04 C/yr)
    t_norm = (years - 1975) / 10.0
    true_mu = 40.0 + 0.45 * t_norm
    ams = true_mu + np.random.gumbel(loc=0.0, scale=1.2, size=len(years))

    ns_gev = NonStationaryGEV()
    summary = ns_gev.fit(years, ams)

    assert ns_gev.fitted is True
    # Captured trend should be positive
    assert ns_gev.mu_1 > 0.20
    # Future projected 50-year return level must increase over time
    rl_2030 = ns_gev.predict_return_level(50, 2030)
    rl_2050 = ns_gev.predict_return_level(50, 2050)
    assert rl_2050 > rl_2030


def test_peaks_over_threshold_declustering_and_gpd():
    """Verifies run-length declustering and GPD return level monotonic scaling."""
    np.random.seed(42)
    # Simulate 15 years of daily rainfall with clustered convective storms
    n_days = 15 * 365
    daily_rain = np.random.exponential(6.0, n_days)
    # Inject storm clusters
    daily_rain[100:103] = [45.0, 85.0, 55.0]  # Cluster 1 (peak 85.0)
    daily_rain[250:252] = [60.0, 95.0]        # Cluster 2 (peak 95.0)
    daily_rain[500:504] = [70.0, 110.0, 65.0, 50.0] # Cluster 3 (peak 110.0)

    pot = PeaksOverThresholdGPD(run_length_days=3)
    peaks, indices = pot.decluster_events(daily_rain, threshold=35.0)

    # All declustered peaks must exceed threshold
    assert np.all(peaks > 35.0)
    assert len(peaks) >= 3
    # Peaks from injected clusters must be present
    assert 85.0 in peaks or 95.0 in peaks or 110.0 in peaks

    # Fit GPD with threshold 22.0
    fit_res = pot.fit(daily_rain, threshold=22.0, total_years=15.0)
    assert pot.fitted is True
    assert pot.sigma > 0.0

    t10 = pot.predict_return_level(10)
    t50 = pot.predict_return_level(50)
    t100 = pot.predict_return_level(100)

    # Monotonic scaling: T100 > T50 > T10
    assert t100 > t50 > t10


def test_drought_cascade_stability_audit():
    """Verifies that DroughtCascade causal forward filtering never underflows."""
    audit = audit_drought_cascade_forward_stability(n_steps=1500)
    assert audit["status"] == "PASSED"
    assert audit["is_non_negative"] is True
    assert audit["is_exact_sum_to_one"] is True
    assert audit["is_finite_no_underflow"] is True
    assert audit["min_probability"] >= 0.0
