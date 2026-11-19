"""
Continuous Drift Detection & Statistical Divergence Engine for Clima-Core (Phase 5).
Implements:
  1. Population Stability Index (PSI) for continuous meteorological and grid features.
  2. Two-Sample Kolmogorov-Smirnov (KS) test for distribution shift detection.
  3. Feature missingness and upstream telemetry availability monitors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from scipy import stats

from climacore.common.logging import get_logger

logger = get_logger("climacore.monitoring.drift")


@dataclass
class DriftMetricResult:
    """Standardized report for a single feature's drift metrics."""
    feature_name: str
    psi_value: float
    psi_status: str  # "STABLE", "MODERATE_DRIFT", "SIGNIFICANT_DRIFT"
    ks_statistic: float
    ks_pvalue: float
    ks_drift_detected: bool
    missing_rate_ref: float
    missing_rate_prod: float
    reference_mean: float
    production_mean: float
    reference_std: float
    production_std: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "psi_value": round(self.psi_value, 5),
            "psi_status": self.psi_status,
            "ks_statistic": round(self.ks_statistic, 5),
            "ks_pvalue": round(self.ks_pvalue, 5),
            "ks_drift_detected": self.ks_drift_detected,
            "missing_rate_ref": round(self.missing_rate_ref, 4),
            "missing_rate_prod": round(self.missing_rate_prod, 4),
            "reference_mean": round(self.reference_mean, 3),
            "production_mean": round(self.production_mean, 3),
            "reference_std": round(self.reference_std, 3),
            "production_std": round(self.production_std, 3),
        }


def calculate_missing_rate(values: np.ndarray) -> float:
    """Calculate the fraction of NaN or missing values in an array."""
    if len(values) == 0:
        return 0.0
    return float(np.isnan(values).sum() / len(values))


def calculate_psi(
    reference: np.ndarray,
    production: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> Tuple[float, str]:
    """
    Calculate the Population Stability Index (PSI) between reference and production distributions:
      PSI = sum((Actual_i - Expected_i) * ln(Actual_i / Expected_i))

    Thresholds:
      PSI < 0.10: "STABLE" (no significant distribution change)
      0.10 <= PSI < 0.25: "MODERATE_DRIFT" (warning: monitor and prepare recalibration)
      PSI >= 0.25: "SIGNIFICANT_DRIFT" (critical: model retraining or recalibration required)
    """
    clean_ref = reference[~np.isnan(reference)]
    clean_prod = production[~np.isnan(production)]

    if len(clean_ref) == 0 or len(clean_prod) == 0:
        return 0.0, "STABLE"

    # Define quantile bin edges from the reference distribution
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(clean_ref, quantiles)

    # Ensure strictly increasing edges to prevent duplicate bins on discrete/constant data
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    for i in range(1, len(bin_edges) - 1):
        if bin_edges[i] <= bin_edges[i - 1]:
            bin_edges[i] = bin_edges[i - 1] + 1e-5

    # Compute frequency counts in each bin
    ref_counts, _ = np.histogram(clean_ref, bins=bin_edges)
    prod_counts, _ = np.histogram(clean_prod, bins=bin_edges)

    # Convert to proportions and apply Laplace smoothing
    ref_props = ref_counts / len(clean_ref)
    prod_props = prod_counts / len(clean_prod)

    ref_props = np.maximum(ref_props, epsilon)
    prod_props = np.maximum(prod_props, epsilon)

    # Normalize proportions so they sum to 1
    ref_props /= ref_props.sum()
    prod_props /= prod_props.sum()

    # Compute PSI sum
    psi_terms = (prod_props - ref_props) * np.log(prod_props / ref_props)
    psi_value = float(np.sum(psi_terms))

    if psi_value < 0.10:
        status = "STABLE"
    elif psi_value < 0.25:
        status = "MODERATE_DRIFT"
    else:
        status = "SIGNIFICANT_DRIFT"

    return psi_value, status


def calculate_ks_drift(
    reference: np.ndarray,
    production: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float, bool]:
    """
    Compute two-sample Kolmogorov-Smirnov test to detect distribution shift:
      Returns: (ks_statistic, p_value, is_drift_detected)
    """
    clean_ref = reference[~np.isnan(reference)]
    clean_prod = production[~np.isnan(production)]

    if len(clean_ref) == 0 or len(clean_prod) == 0:
        return 0.0, 1.0, False

    stat, pval = stats.ks_2samp(clean_ref, clean_prod)
    is_drift = bool(pval < alpha)
    return float(stat), float(pval), is_drift


def evaluate_feature_drift(
    feature_name: str,
    reference_data: np.ndarray,
    production_data: np.ndarray,
    num_bins: int = 10,
    alpha: float = 0.05,
) -> DriftMetricResult:
    """Comprehensive evaluation of feature distribution stability."""
    ref_arr = np.asarray(reference_data, dtype=float)
    prod_arr = np.asarray(production_data, dtype=float)

    miss_ref = calculate_missing_rate(ref_arr)
    miss_prod = calculate_missing_rate(prod_arr)

    clean_ref = ref_arr[~np.isnan(ref_arr)]
    clean_prod = prod_arr[~np.isnan(prod_arr)]

    psi_val, psi_stat = calculate_psi(clean_ref, clean_prod, num_bins=num_bins)
    ks_stat, ks_pval, ks_drift = calculate_ks_drift(clean_ref, clean_prod, alpha=alpha)

    ref_m = float(np.mean(clean_ref)) if len(clean_ref) > 0 else 0.0
    prod_m = float(np.mean(clean_prod)) if len(clean_prod) > 0 else 0.0
    ref_s = float(np.std(clean_ref)) if len(clean_ref) > 0 else 0.0
    prod_s = float(np.std(clean_prod)) if len(clean_prod) > 0 else 0.0

    return DriftMetricResult(
        feature_name=feature_name,
        psi_value=psi_val,
        psi_status=psi_stat,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pval,
        ks_drift_detected=ks_drift,
        missing_rate_ref=miss_ref,
        missing_rate_prod=miss_prod,
        reference_mean=ref_m,
        production_mean=prod_m,
        reference_std=ref_s,
        production_std=prod_s,
    )
