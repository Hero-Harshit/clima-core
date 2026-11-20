"""Unit tests for CompoundRisk V3 Regular Vine Copulas:
- Bivariate pair-copula density, h-functions, and maximum likelihood estimation
- 3-Variate D-Vine decomposition for Deluge-Hydrodynamic Cascade (P -> SM -> Q)
- Joint compound exceedance probability and Risk Amplification Factor
"""

import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import pytest
import numpy as np

from src.climacore.analytics.vine_copulas import PairCopula, DelugeHydrodynamicVine


def test_pair_copula_families_pdf_and_h_function():
    """Tests density positivity and h-function probability bounds across copula families."""
    u = np.linspace(0.05, 0.95, 10)
    v = np.linspace(0.05, 0.95, 10)

    families = [
        ("gaussian", 0.6),
        ("clayton", 1.8),
        ("gumbel", 2.2),
        ("frank", 4.0),
        ("independence", 0.0),
    ]

    for fam, th in families:
        copula = PairCopula(family=fam, theta=th)
        pdf_val = copula.pdf(u, v)
        h_val = copula.h_function(u, v)

        assert np.all(pdf_val > 0.0), f"PDF failed for {fam}"
        assert np.all(h_val >= 0.0), f"h_function lower bound failed for {fam}"
        assert np.all(h_val <= 1.0), f"h_function upper bound failed for {fam}"


def test_pair_copula_mle_fitting():
    """Tests automated maximum likelihood selection and parameter estimation."""
    np.random.seed(42)
    # Generate correlated bivariate Gaussian pseudo-observations (rho = 0.7)
    mean = [0.0, 0.0]
    cov = [[1.0, 0.7], [0.7, 1.0]]
    raw = np.random.multivariate_normal(mean, cov, 300)
    u = np.argsort(np.argsort(raw[:, 0])) / 301.0
    v = np.argsort(np.argsort(raw[:, 1])) / 301.0

    fitted = PairCopula.fit(u, v)
    assert fitted.family in ("gaussian", "gumbel", "frank")
    assert fitted.theta > 0.3


def test_deluge_hydrodynamic_d_vine():
    """Tests 3-variate D-vine cascade calibration, simulation, and joint compound risk."""
    np.random.seed(42)
    n = 250

    # Simulate physical cascade: Rain P -> Soil Moisture SM -> Streamflow Q
    prcp = np.random.exponential(15.0, n)
    soil_moisture = 0.4 * np.roll(prcp, 1) + 0.5 * prcp + np.random.normal(20.0, 3.0, n)
    streamflow = 0.6 * soil_moisture + 0.3 * prcp + np.random.exponential(5.0, n)

    vine = DelugeHydrodynamicVine()
    vine.fit(prcp, soil_moisture, streamflow)

    assert vine.fitted is True
    assert vine.c_p_sm is not None
    assert vine.c_sm_q is not None
    assert vine.c_p_q_given_sm is not None

    # Evaluate joint log-likelihood
    u_p = np.argsort(np.argsort(prcp)) / (n + 1.0)
    u_sm = np.argsort(np.argsort(soil_moisture)) / (n + 1.0)
    u_q = np.argsort(np.argsort(streamflow)) / (n + 1.0)
    ll = vine.log_likelihood(u_p, u_sm, u_q)
    assert not np.isnan(ll)
    assert not np.isinf(ll)

    # Evaluate joint compound risk at 85th percentile
    res = vine.evaluate_joint_compound_risk(quantile_p=0.85, quantile_sm=0.85, quantile_q=0.85, n_mc=5000)
    assert "p_compound_vine" in res
    assert "p_independent" in res
    assert res["p_compound_vine"] > 0.0
    # Positively correlated physical deluge cascade must exhibit risk amplification > 1.0
    assert res["risk_amplification_factor"] > 1.0
