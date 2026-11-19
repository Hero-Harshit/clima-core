"""
Continuous Model Monitoring & Health Service for Clima-Core (Phase 5).
Coordinates:
  1. Feature Drift Detection across continuous telemetry features.
  2. Conformal Calibration Health & Exchangeability Degradation Tracking.
  3. Latency Profiling (P50, P95, P99) and inference throughput.
  4. System Health Diagnostic reporting for operations and alerting.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Dict, List, Any, Optional
import numpy as np

from climacore.common.logging import get_logger
from climacore.monitoring.drift import evaluate_feature_drift, DriftMetricResult

logger = get_logger("climacore.monitoring.service")


@dataclass
class CalibrationRecord:
    """A single verified observation against a conformal prediction interval."""
    timestamp: str
    target_name: str
    y_true: float
    interval_lower: float
    interval_upper: float
    covered: bool
    interval_width: float


class CalibrationHealthMonitor:
    """
    Monitors split conformal calibration coverage over rolling prediction batches.
    Detects exchangeability violations and distribution shifts that degrade finite-sample coverage.
    """

    def __init__(self, target_coverage: float = 0.90, rolling_window_size: int = 100, alert_threshold: float = 0.85):
        self.target_coverage = target_coverage
        self.rolling_window_size = rolling_window_size
        self.alert_threshold = alert_threshold
        self._history: deque[CalibrationRecord] = deque(maxlen=rolling_window_size)

    def record_verification(
        self,
        target_name: str,
        y_true: float,
        interval_lower: float,
        interval_upper: float,
    ) -> bool:
        """Record an observed outcome against the conformal bounds."""
        covered = bool(interval_lower <= y_true <= interval_upper)
        width = float(interval_upper - interval_lower)
        record = CalibrationRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            target_name=target_name,
            y_true=float(y_true),
            interval_lower=float(interval_lower),
            interval_upper=float(interval_upper),
            covered=covered,
            interval_width=width,
        )
        self._history.append(record)
        return covered

    def get_health_status(self) -> Dict[str, Any]:
        """Compute current rolling coverage and calibration status."""
        total = len(self._history)
        if total == 0:
            return {
                "status": "INITIALIZING",
                "sample_count": 0,
                "target_coverage": self.target_coverage,
                "empirical_coverage": None,
                "mean_interval_width": None,
                "alert_active": False,
            }

        covered_count = sum(1 for r in self._history if r.covered)
        empirical_cov = covered_count / total
        mean_width = float(np.mean([r.interval_width for r in self._history]))
        alert_active = bool(total >= 20 and empirical_cov < self.alert_threshold)

        return {
            "status": "CALIBRATION_DRIFT_ALERT" if alert_active else "HEALTHY",
            "sample_count": total,
            "target_coverage": self.target_coverage,
            "empirical_coverage": round(empirical_cov, 4),
            "mean_interval_width": round(mean_width, 4),
            "alert_active": alert_active,
            "recommendation": (
                "TRIGGER_CONFORMAL_RECALIBRATION: Exchangeability violation detected"
                if alert_active
                else "CALIBRATION_NOMINAL: Bounds adhere to coverage guarantees"
            ),
        }


class LatencyProfiler:
    """Tracks API and model inference execution times across percentiles."""

    def __init__(self, max_samples: int = 500):
        self._latencies_ms: deque[float] = deque(maxlen=max_samples)
        self._error_count: int = 0
        self._total_requests: int = 0

    def record(self, latency_ms: float, is_error: bool = False):
        self._latencies_ms.append(float(latency_ms))
        self._total_requests += 1
        if is_error:
            self._error_count += 1

    def get_summary(self) -> Dict[str, Any]:
        if not self._latencies_ms:
            return {
                "total_requests": self._total_requests,
                "error_count": self._error_count,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "mean_ms": 0.0,
            }

        arr = np.array(self._latencies_ms)
        return {
            "total_requests": self._total_requests,
            "error_count": self._error_count,
            "error_rate": round(self._error_count / max(1, self._total_requests), 4),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "mean_ms": round(float(np.mean(arr)), 2),
            "recent_sample_size": len(arr),
        }


class ModelMonitoringService:
    """
    Central continuous monitoring service for Clima-Core.
    Tracks feature distribution drift, missingness, calibration degradation, and system latency.
    """

    def __init__(self):
        self.calibration_monitor = CalibrationHealthMonitor(target_coverage=0.90)
        self.latency_profiler = LatencyProfiler()
        self._reference_distributions: Dict[str, np.ndarray] = {}
        self._initialize_default_baselines()

    def _initialize_default_baselines(self):
        """Populate initial representative baselines for key environmental features."""
        np.random.seed(42)
        # Representative distributions based on historical Indian climatology
        self._reference_distributions = {
            "temperature_max": np.random.normal(loc=32.5, scale=6.0, size=500),
            "rainfall_mm": np.random.exponential(scale=12.0, size=500),
            "peak_grid_load_mw": np.random.normal(loc=12500.0, scale=2500.0, size=500),
            "pm25_concentration": np.random.lognormal(mean=3.8, sigma=0.8, size=500),
            "soil_saturation_ratio": np.random.beta(a=2.0, b=3.0, size=500),
            "wind_speed_kmh": np.random.gamma(shape=3.0, scale=4.0, size=500),
        }

    def set_feature_baseline(self, feature_name: str, baseline_data: np.ndarray):
        """Set or update the reference distribution for a feature."""
        self._reference_distributions[feature_name] = np.asarray(baseline_data, dtype=float)

    def evaluate_production_batch(
        self,
        production_features: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of streaming production telemetry against registered baselines.
        Returns a complete system health and drift report.
        """
        feature_reports: Dict[str, Any] = {}
        drift_warnings = 0
        significant_drifts = 0

        for feat_name, prod_data in production_features.items():
            if feat_name in self._reference_distributions:
                ref_data = self._reference_distributions[feat_name]
                res = evaluate_feature_drift(feat_name, ref_data, prod_data)
                feature_reports[feat_name] = res.to_dict()

                if res.psi_status == "MODERATE_DRIFT":
                    drift_warnings += 1
                elif res.psi_status == "SIGNIFICANT_DRIFT":
                    significant_drifts += 1
            else:
                feature_reports[feat_name] = {
                    "feature_name": feat_name,
                    "status": "UNREGISTERED_BASELINE",
                    "sample_count": len(prod_data),
                }

        # Overall System Status determination
        calib_health = self.calibration_monitor.get_health_status()
        if significant_drifts > 0 or calib_health.get("alert_active", False):
            overall_status = "CRITICAL_DRIFT"
        elif drift_warnings > 0:
            overall_status = "WARNING_DRIFT"
        else:
            overall_status = "HEALTHY"

        return {
            "monitoring_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "overall_status": overall_status,
            "drift_warnings_count": drift_warnings,
            "significant_drifts_count": significant_drifts,
            "features_analyzed": len(production_features),
            "feature_drift_metrics": feature_reports,
            "calibration_health": calib_health,
            "latency_metrics": self.latency_profiler.get_summary(),
        }


# Global singleton instance for operational usage
_MONITOR_INSTANCE: Optional[ModelMonitoringService] = None


def get_monitoring_service() -> ModelMonitoringService:
    """Retrieve global singleton monitoring service."""
    global _MONITOR_INSTANCE
    if _MONITOR_INSTANCE is None:
        _MONITOR_INSTANCE = ModelMonitoringService()
    return _MONITOR_INSTANCE
