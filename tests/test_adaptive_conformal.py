"""Unit tests for the Shared Adaptive Uncertainty Calibration Engine:
- Adaptive Conformal Inference (ACI) online tracking and drift adaptation
- Rolling-Window Split Conformal interval calibration
- Classification Calibration (Platt & Isotonic)
- Expected Calibration Error (ECE) and reliability diagram binning
"""

import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import pytest
import numpy as np

from src.climacore.uncertainty.adaptive_conformal import (
    AdaptiveConformalInference,
    RollingSplitConformal,
    ClassificationCalibrator,
)


def test_adaptive_conformal_inference_online_tracking():
    """Tests ACI online adaptation under a distribution shift scenario."""
    np.random.seed(42)
    # Target 90% coverage -> alpha = 0.10
    aci = AdaptiveConformalInference(target_alpha=0.10, gamma=0.03, initial_quantile=1.5, window_size=100)

    # Phase 1: Stationary regime (sigma=1.0) for 150 steps
    # Phase 2: Sudden volatility spike / heatwave regime (sigma=2.5) for 150 steps
    n_steps = 300
    y_hat_series = np.sin(np.linspace(0, 10, n_steps)) * 10.0 + 30.0
    noise = np.concatenate([
        np.random.normal(0, 1.0, 150),
        np.random.normal(0, 2.5, 150),
    ])
    y_obs_series = y_hat_series + noise

    for t in range(n_steps):
        step_rec = aci.update(y_obs=y_obs_series[t], y_hat=y_hat_series[t])
        assert step_rec["lower"] < step_rec["upper"]
        assert 0.01 <= step_rec["next_alpha"] <= 0.99

    summary = aci.get_summary()
    assert summary["total_steps"] == 300
    assert summary["target_coverage_pct"] == 90.0
    # Even with a 2.5x variance spike, ACI should maintain long-run coverage within 6% of target
    assert abs(summary["coverage_gap_pct"]) < 6.0
    assert summary["mean_interval_width"] > 0.0


def test_rolling_split_conformal_intervals():
    """Tests RollingSplitConformal finite-sample prediction interval validity."""
    np.random.seed(42)
    calibrator = RollingSplitConformal(target_coverage=0.80, window_size=100)

    y_true_cal = np.random.normal(25.0, 3.0, 150)
    y_pred_cal = y_true_cal + np.random.normal(0.0, 1.0, 150)

    calibrator.fit(y_true_cal, y_pred_cal)
    q = calibrator.get_quantile()
    assert q > 0.0

    y_test_pred = np.array([20.0, 25.0, 30.0, 35.0])
    lower, upper = calibrator.predict_intervals(y_test_pred)

    assert len(lower) == 4
    assert len(upper) == 4
    assert np.all(lower < upper)
    assert np.allclose(upper - lower, 2 * q)

    # Test update sliding window
    calibrator.update(y_obs=28.0, y_pred=25.0)
    assert len(calibrator.residuals) <= 100


def test_classification_calibrator_isotonic_and_platt():
    """Tests Platt and Isotonic calibration reducing ECE on overconfident probabilities."""
    np.random.seed(42)
    n = 1000

    # True underlying probability
    p_true = np.random.uniform(0.05, 0.95, n)
    y_true = (np.random.uniform(0.0, 1.0, n) < p_true).astype(np.float64)

    # Heavily distorted/overconfident uncalibrated predictions
    p_raw = np.where(p_true > 0.5, p_true ** 0.5, p_true ** 2.0)
    p_raw = np.clip(p_raw, 0.01, 0.99)

    raw_metrics = ClassificationCalibrator.compute_calibration_metrics(p_raw, y_true, n_bins=10)
    assert raw_metrics["ece"] > 0.05

    # 1. Isotonic Calibration
    iso_cal = ClassificationCalibrator(method="isotonic")
    iso_cal.fit(p_raw[:600], y_true[:600])
    p_cal_iso = iso_cal.predict_proba(p_raw[600:])
    iso_metrics = ClassificationCalibrator.compute_calibration_metrics(p_cal_iso, y_true[600:], n_bins=10)

    assert np.all(p_cal_iso >= 0.0)
    assert np.all(p_cal_iso <= 1.0)
    assert iso_metrics["ece"] < raw_metrics["ece"]
    assert iso_metrics["brier_score"] <= raw_metrics["brier_score"]

    # 2. Platt Scaling
    platt_cal = ClassificationCalibrator(method="platt")
    platt_cal.fit(p_raw[:600], y_true[:600])
    p_cal_platt = platt_cal.predict_proba(p_raw[600:])
    platt_metrics = ClassificationCalibrator.compute_calibration_metrics(p_cal_platt, y_true[600:], n_bins=10)

    assert np.all(p_cal_platt >= 0.0)
    assert np.all(p_cal_platt <= 1.0)
    assert platt_metrics["ece"] < raw_metrics["ece"]


def test_compute_calibration_metrics_perfect_and_pathological():
    """Tests calibration metrics under edge-case conditions."""
    # Perfect calibration
    p_perf = np.array([0.1, 0.1, 0.9, 0.9])
    y_perf = np.array([0.0, 0.0, 1.0, 1.0])
    perf_metrics = ClassificationCalibrator.compute_calibration_metrics(p_perf, y_perf, n_bins=5)
    assert perf_metrics["ece"] < 0.15
    assert len(perf_metrics["reliability_diagram_bins"]) == 5

    # Empty inputs
    empty_metrics = ClassificationCalibrator.compute_calibration_metrics(np.array([]), np.array([]))
    assert empty_metrics["ece"] == 0.0
    assert empty_metrics["brier_score"] == 0.0
