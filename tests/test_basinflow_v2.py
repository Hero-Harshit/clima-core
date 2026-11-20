"""Unit tests for BasinFlow V2:
- Hydrological evaluation metrics (NSE, KGE, LogNSE, FHV, FLV, WB)
- Conceptual models (GR4J, HBV-Light)
- Physical mass conservation invariant in MC-LSTM and MC-LSTM-DS
- Unit hydrograph and routing baselines
"""

import sys
from pathlib import Path

# Ensure root directory is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import pytest
import numpy as np
import torch
import pandas as pd

from src.climacore.models.basinflow.metrics import compute_hydrological_metrics
from src.climacore.models.basinflow.conceptual_models import GR4J, HBVLight
from src.climacore.models.basinflow.baselines import (
    PersistenceBaseline,
    SeasonalClimatologyBaseline,
    SCSCNBaseline,
    BasinFlowV1Baseline,
)
from src.climacore.models.basinflow.neural_hydrology import (
    StandardLSTM,
    MCLSTM,
    MCLSTMDualState,
)


def test_hydrological_metrics_perfect_score():
    """Identical observed and simulated streamflows should yield NSE=1, KGE=1, WB=0."""
    q_obs = np.array([0.5, 1.2, 5.0, 12.4, 25.0, 8.2, 2.1, 0.8, 0.4, 0.2], dtype=np.float64)
    q_sim = q_obs.copy()

    metrics = compute_hydrological_metrics(q_obs, q_sim)
    assert np.isclose(metrics["nse"], 1.0, atol=1e-3)
    assert np.isclose(metrics["kge"], 1.0, atol=1e-3)
    assert np.isclose(metrics["rmse"], 0.0, atol=1e-3)
    assert np.isclose(metrics["water_balance_err_pct"], 0.0, atol=1e-2)
    assert metrics["peak_timing_err_days"] == 0.0


def test_hydrological_metrics_handles_nans():
    """Metrics must filter out NaNs and negative values safely."""
    q_obs = np.array([np.nan, 2.0, 4.0, -99.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0])
    q_sim = np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0])

    metrics = compute_hydrological_metrics(q_obs, q_sim)
    assert metrics["nse"] > 0.99
    assert metrics["kge"] > 0.99


def test_gr4j_simulation_and_serialization():
    """Tests GR4J forward simulation, store non-negativity, and dictionary roundtrip."""
    p = np.array([0.0, 5.0, 25.0, 40.0, 15.0, 2.0, 0.0, 0.0, 1.0, 0.0])
    e = np.array([4.0, 4.0, 3.0, 2.0, 3.0, 4.0, 4.5, 5.0, 4.0, 4.0])

    gr = GR4J(x1=300.0, x2=0.5, x3=80.0, x4=1.8)
    res = gr.simulate(p, e, return_stores=True)

    q = res["q_sim"]
    s = res["s_store"]
    r = res["r_store"]

    assert len(q) == len(p)
    assert np.all(q >= 0.0)
    assert np.all(s >= 0.0)
    assert np.all(r >= 0.0)
    assert np.all(s <= 300.0)

    # Serialization test
    d = gr.to_dict()
    gr_loaded = GR4J.from_dict(d)
    assert np.allclose(gr.params, gr_loaded.params)


def test_hbv_light_simulation_and_serialization():
    """Tests HBV-Light forward simulation, non-negative storage, and dictionary roundtrip."""
    p = np.array([0.0, 10.0, 30.0, 50.0, 20.0, 5.0, 0.0, 0.0, 2.0, 0.0])
    e = np.array([3.0, 3.0, 2.5, 2.0, 2.5, 3.0, 3.5, 4.0, 3.5, 3.0])

    hbv = HBVLight(fc=250.0, beta=2.0, lp=0.7, perc=1.5, uzl=25.0, k0=0.15, k1=0.08, k2=0.02, maxbas=2.5)
    res = hbv.simulate(p, e, return_stores=True)

    q = res["q_sim"]
    assert len(q) == len(p)
    assert np.all(q >= 0.0)
    assert np.all(res["soil_store"] >= 0.0)
    assert np.all(res["upper_store"] >= 0.0)
    assert np.all(res["lower_store"] >= 0.0)

    # Serialization test
    d = hbv.to_dict()
    hbv_loaded = HBVLight.from_dict(d)
    assert np.allclose(hbv.params, hbv_loaded.params)


def test_mc_lstm_mass_conservation_invariant():
    """Tests that MC-LSTM mathematically conserves precipitation mass at every step."""
    torch.manual_seed(42)
    mc = MCLSTM(input_dim=5, hidden_dim=8)
    mc.eval()

    batch_size = 2
    seq_len = 20
    x = torch.randn(batch_size, seq_len, 5)
    # Strictly positive precipitation
    raw_p = torch.abs(torch.randn(batch_size, seq_len)) * 10.0

    c_t = torch.zeros(batch_size, 8)
    total_inflow = torch.zeros(batch_size)
    total_released = torch.zeros(batch_size)

    for t in range(seq_len):
        x_t = x[:, t, :]
        p_t = raw_p[:, t:t+1]
        c_prev = c_t.clone()

        c_t, _, rel_t = mc.cell(x_t, p_t, c_prev)

        # Invariant at each step: sum(c_t) = sum(c_{t-1}) + p_t - sum(rel_t)
        delta_c = torch.sum(c_t, dim=-1) - torch.sum(c_prev, dim=-1)
        expected_delta = p_t.squeeze(-1) - torch.sum(rel_t, dim=-1)
        
        # Discrepancy must be machine-epsilon zero
        assert torch.allclose(delta_c, expected_delta, atol=1e-5), f"Mass violation at step {t}: {delta_c} vs {expected_delta}"

        total_inflow += p_t.squeeze(-1)
        total_released += torch.sum(rel_t, dim=-1)

    # Global mass balance: sum(P) = sum(Released) + Final Storage
    final_storage = torch.sum(c_t, dim=-1)
    mass_balance_error = torch.abs(total_inflow - (total_released + final_storage))
    assert torch.all(mass_balance_error < 1e-4), f"Cumulative mass error: {mass_balance_error}"


def test_mc_lstm_dual_state_flow_separation():
    """Tests MC-LSTM-DS quick/slow flow separation and non-negativity."""
    torch.manual_seed(42)
    mcds = MCLSTMDualState(input_dim=5, reservoir_dim=8)
    mcds.eval()

    batch_size = 2
    seq_len = 15
    x = torch.randn(batch_size, seq_len, 5)
    raw_p = torch.abs(torch.randn(batch_size, seq_len)) * 15.0

    final_q, diagnostics = mcds(x, raw_p)

    q_tot = diagnostics["q_total_series"]
    q_quick = diagnostics["q_quick_series"]
    q_slow = diagnostics["q_slow_series"]

    # 1. Total flow must equal quickflow + slowflow
    assert torch.allclose(q_tot, q_quick + q_slow, atol=1e-5)

    # 2. Both streams must be non-negative
    assert torch.all(q_quick >= 0.0)
    assert torch.all(q_slow >= 0.0)
    assert torch.all(final_q >= 0.0)


def test_baselines():
    """Tests Persistence, Seasonal Climatology, and SCS-CN baselines."""
    dates = pd.date_range("2010-01-01", periods=10, freq="D")
    q_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    pers = PersistenceBaseline()
    pers.fit(q_train)
    q_pred = pers.predict(q_train)
    assert len(q_pred) == 10
    assert q_pred[1] == 1.0

    seas = SeasonalClimatologyBaseline()
    seas.fit(dates, q_train)
    s_pred = seas.predict(dates)
    assert len(s_pred) == 10
    assert np.all(s_pred > 0.0)

    scs = SCSCNBaseline(cn=80.0)
    p = np.array([0.0, 10.0, 50.0])
    q_scs = scs.predict(p)
    assert q_scs[0] == 0.0
    assert q_scs[2] > q_scs[1]
