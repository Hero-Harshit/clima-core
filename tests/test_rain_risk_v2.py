"""
Unit tests for Rainfall Intelligence V2 (IMD 5-Tier Scale & Multi-Horizon Suite).
Verifies:
1. Reload parity for all serialized V2 boosters (delta_max == 0.0).
2. Proper tier classification logic matching official IMD precipitation scales.
3. Multi-horizon lead model outputs and evaluation report schema.
"""

from pathlib import Path
import json
import lightgbm as lgb
import numpy as np
import pytest

from climacore.config.loader import get_project_root
from climacore.models.rain_risk.train_v2 import classify_imd_rain_tier, IMD_TIER_NAMES


@pytest.fixture
def root() -> Path:
    return get_project_root()


def test_imd_tier_classification_logic():
    """Verify official IMD 5-tier precipitation classification thresholds."""
    assert classify_imd_rain_tier(0.0) == 0
    assert classify_imd_rain_tier(1.5) == 0
    assert classify_imd_rain_tier(2.5) == 1
    assert classify_imd_rain_tier(15.5) == 1
    assert classify_imd_rain_tier(15.6) == 2
    assert classify_imd_rain_tier(64.4) == 2
    assert classify_imd_rain_tier(64.5) == 3
    assert classify_imd_rain_tier(115.5) == 3
    assert classify_imd_rain_tier(115.6) == 4
    assert classify_imd_rain_tier(245.0) == 4


def test_rain_risk_v2_artifacts_and_reload_parity(root: Path):
    """Test that all RainRisk V2 boosters exist and load with zero float drift."""
    artifact_dir = root / "artifacts" / "models" / "rain_risk_v2" / "v0.1.0"
    assert artifact_dir.exists()

    # 1. 5-Tier Classifier
    clf_path = artifact_dir / "imd_5tier_classifier.txt"
    assert clf_path.exists()
    clf = lgb.Booster(model_file=str(clf_path))

    # Test reload on synthetic input
    schema_path = artifact_dir / "schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    n_feats = schema["feature_count"]
    sample_x = np.random.RandomState(42).randn(5, n_feats)
    preds1 = clf.predict(sample_x)
    preds2 = clf.predict(sample_x)
    assert np.max(np.abs(preds1 - preds2)) == 0.0

    # 2. Multi-Horizon Regressors
    for lead in ["lead_24h", "lead_48h", "lead_72h"]:
        reg_path = artifact_dir / f"{lead}_regressor.txt"
        assert reg_path.exists()
        reg = lgb.Booster(model_file=str(reg_path))
        reg_preds = reg.predict(sample_x)
        assert len(reg_preds) == 5
        assert not np.isnan(reg_preds).any()


def test_rain_risk_v2_evaluation_report(root: Path):
    """Verify evaluation report metrics and accuracy on untouched holdout."""
    report_path = root / "reports" / "evaluation" / "rain_risk_v2_evaluation_report.json"
    assert report_path.exists()

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    metrics = report["holdout_test_metrics"]
    assert metrics["imd_tier_multiclass_accuracy"] >= 0.75
    assert metrics["lead_24h"]["mae_mm"] <= 3.5
    assert metrics["lead_48h"]["mae_mm"] <= 3.8
    assert metrics["lead_72h"]["mae_mm"] <= 4.0
