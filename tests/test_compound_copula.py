"""Unit tests for Compound Copula Engine V1."""

import numpy as np
import pytest

from climacore.analytics.copula import (
    CompoundCopulaEngine,
    CopulaFamily,
    JointReturnPeriodResult,
    compute_pseudo_observations,
)


def test_pseudo_observations_range():
    np.random.seed(42)
    x = np.random.normal(size=50)
    y = np.random.normal(size=50)
    u, v = compute_pseudo_observations(x, y)

    assert len(u) == 50
    assert len(v) == 50
    assert np.all(u > 0.0) and np.all(u < 1.0)
    assert np.all(v > 0.0) and np.all(v < 1.0)


def test_candidate_families_fitting_and_selection():
    np.random.seed(42)
    # Generate upper-tail correlated data (Gumbel-like)
    x = np.random.exponential(scale=2.0, size=80)
    y = 0.7 * x + 0.3 * np.random.exponential(scale=2.0, size=80)

    engine = CompoundCopulaEngine(bootstrap_iterations=30)
    u, v = compute_pseudo_observations(x, y)
    candidates = engine.fit_candidate_families(u, v)

    assert len(candidates) == 4
    families = [c.family for c in candidates]
    assert CopulaFamily.GUMBEL in families
    assert CopulaFamily.CLAYTON in families
    assert CopulaFamily.SURVIVAL_CLAYTON in families
    assert CopulaFamily.FRANK in families

    # Sorted by AIC ascending
    assert candidates[0].aic <= candidates[1].aic <= candidates[2].aic <= candidates[3].aic
    assert candidates[0].theta > 0.0


def test_joint_return_periods_properties():
    np.random.seed(42)
    x = np.random.normal(loc=40.0, scale=3.0, size=100)
    y = 0.5 * x + np.random.normal(loc=50.0, scale=10.0, size=100)

    engine = CompoundCopulaEngine(bootstrap_iterations=40)
    # 10-year marginal return period (quantile = 0.90)
    res = engine.evaluate_joint_return_periods(
        series_x=x,
        series_y=y,
        target_quantile_x=0.90,
        target_quantile_y=0.90,
        hazard_x_name="Heatwave_Max_Temp",
        hazard_y_name="Concurrent_Peak_Precip",
    )

    assert isinstance(res, JointReturnPeriodResult)
    assert res.marginal_return_period_x_years == 10.0
    assert res.marginal_return_period_y_years == 10.0

    # Fundamental mathematical relations:
    # 1. Concurrent AND event is rarer than univariate event: T_AND > Tx
    assert res.t_and_years > 10.0

    # 2. Union OR event is more frequent than univariate event: T_OR < Tx
    assert res.t_or_years < 10.0

    # 3. Probabilities satisfy: P_AND < (1 - u) and P_OR > (1 - u)
    assert res.p_and < 0.10
    assert res.p_or > 0.10

    # 4. Bootstrap 95% CI covers T_AND
    ci_low, ci_high = res.bootstrap_ci_95_t_and
    assert ci_low <= res.t_and_years <= ci_high
    assert res.event_scope == "annual_maxima_coincidence"
    assert res.temporal_resolution == "annual"
    assert "Confidence intervals must be considered" in res.scientific_disclaimer
