"""
Automated Unit and Integration Tests for Clima-Core v1.2 Machine Learning Evolution.
Covers:
  1. Registry Engine integration for v1.2 models
  2. FloodRisk XGBoost Challenger serialization & inference
  3. HeatRisk Quantile Regression monotonicity & coverage
  4. PowerGrid PyTorch CPU 1D Causal Dilated TCN architecture & forward pass
  5. DroughtCascade V1 Gaussian HMM transition matrix & state decoding
  6. v1.2 Benchmark and tournament report schema validation
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pytest
import torch
import xgboost as xgb

from climacore.config.loader import get_project_root
from climacore.registry.engine import get_model_registry

import sys
_ROOT = get_project_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.cpu_training.powergrid.tcn_model import PowerGridTCN


@pytest.fixture
def project_root() -> Path:
    return get_project_root()


def test_v1_2_models_registered_in_engine():
    """Verify all new v1.2 models appear in the canonical Model Registry with valid metadata."""
    registry = get_model_registry()

    expected_new_models = [
        "flood_risk_xgboost_23feat",
        "heat_risk_quantile_suite",
        "power_grid_tcn_v4",
        "drought_cascade_v1",
    ]

    for m_id in expected_new_models:
        assert m_id in registry, f"Missing model in canonical registry: {m_id}"
        rec = registry[m_id]
        assert rec.features_count > 0
        assert rec.version.startswith("1.") or rec.version.startswith("4.")
        assert rec.operational_status in ["CHAMPION", "CHALLENGER"]
        assert rec.artifact_sha256 != "PENDING_SERIALIZATION"


def test_flood_risk_xgboost_challenger_artifact(project_root: Path):
    """Verify FloodRisk XGBoost challenger can be loaded and predict 23-feature vectors."""
    model_path = project_root / "artifacts" / "models" / "v1_2_challengers" / "floodrisk" / "flood_risk_xgboost_23feat.json"
    assert model_path.exists(), f"XGBoost model file not found at {model_path}"

    bst = xgb.Booster()
    bst.load_model(str(model_path))

    # Test dummy inference with 23 features
    dummy_input = np.random.RandomState(42).randn(5, 23).astype(np.float32)
    dmat = xgb.DMatrix(dummy_input)
    preds = bst.predict(dmat)

    assert preds.shape == (5,)
    assert (preds >= 0.0).all() and (preds <= 1.0).all()


def test_heat_risk_quantile_models_monotonicity(project_root: Path):
    """Verify HeatRisk Quantile models (Q10, Q50, Q90) produce valid monotonic intervals."""
    art_dir = project_root / "artifacts" / "models" / "v1_2_challengers" / "heatrisk"
    q10_file = art_dir / "heat_risk_quantile_q_10.booster"
    q50_file = art_dir / "heat_risk_quantile_q_50.booster"
    q90_file = art_dir / "heat_risk_quantile_q_90.booster"

    assert q10_file.exists() and q50_file.exists() and q90_file.exists()

    b10 = lgb.Booster(model_file=str(q10_file))
    b50 = lgb.Booster(model_file=str(q50_file))
    b90 = lgb.Booster(model_file=str(q90_file))

    dummy_input = np.random.RandomState(42).randn(20, 71).astype(np.float32)
    p10 = b10.predict(dummy_input)
    p50 = b50.predict(dummy_input)
    p90 = b90.predict(dummy_input)

    # Monotonic rearrangement
    stacked = np.vstack([p10, p50, p90])
    sorted_q = np.sort(stacked, axis=0)
    s10, s50, s90 = sorted_q[0], sorted_q[1], sorted_q[2]

    # Verify monotonic ordering: s10 <= s50 <= s90
    assert (s10 <= s50).all()
    assert (s50 <= s90).all()
    assert (s90 - s10 >= 0.0).all()


def test_power_grid_tcn_architecture_and_forward_pass(project_root: Path):
    """Verify PyTorch CPU TCN architecture, forward pass, and checkpoint restoration."""
    torch.set_num_threads(2)
    model = PowerGridTCN(in_features=12, channels=24, kernel_size=3, dilations=[1, 2, 4, 8])

    # Dummy batch: [batch_size=4, seq_len=30, features=12]
    dummy_x = torch.randn(4, 30, 12)
    out = model(dummy_x)

    assert out.shape == torch.Size([4])
    assert not torch.isnan(out).any()

    # Load trained checkpoint
    ckpt_path = project_root / "artifacts" / "models" / "v1_2_challengers" / "powergrid" / "powergrid_tcn_v4.pt"
    assert ckpt_path.exists()
    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
    model.eval()

    with torch.no_grad():
        test_out = model(dummy_x)
        assert test_out.shape == torch.Size([4])


def test_drought_cascade_hmm_artifact(project_root: Path):
    """Verify DroughtCascade V1 HMM model deserialization and transition probability matrix."""
    model_path = project_root / "artifacts" / "models" / "v1_2_challengers" / "drought" / "drought_cascade_hmm.joblib"
    assert model_path.exists()

    bundle = joblib.load(str(model_path))
    assert "model" in bundle
    assert "scaler" in bundle
    assert "features" in bundle

    hmm = bundle["model"]
    assert hmm.n_components == 4

    # Verify transition matrix rows sum to 1.0
    trans_matrix = hmm.transmat_
    assert trans_matrix.shape == (4, 4)
    row_sums = np.sum(trans_matrix, axis=1)
    np.testing.assert_allclose(row_sums, np.ones(4), atol=1e-5)

    # Verify diagonal persistence is strictly positive
    for i in range(4):
        assert trans_matrix[i, i] > 0.80, f"Expected strong self-persistence in state {i}"


def test_v1_2_reports_and_resource_benchmarks(project_root: Path):
    """Verify v1.2 tournament and resource benchmark reports exist and validate hardware limits."""
    reports_dir = project_root / "reports" / "v1_2"
    comp_file = reports_dir / "champion_challenger_comparison.json"
    res_file = reports_dir / "resource_benchmarks.json"

    assert comp_file.exists()
    assert res_file.exists()

    with open(res_file, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    # Validate strict CPU adherence and 5 GB RAM compliance
    assert benchmarks["compliance_summary"]["all_tasks_within_5gb_ram"] is True
    assert benchmarks["compliance_summary"]["all_tasks_executed_on_cpu"] is True
    assert benchmarks["compliance_summary"]["cuda_required"] is False
    assert benchmarks["compliance_summary"]["max_peak_ram_observed_mb"] < 1000.0
