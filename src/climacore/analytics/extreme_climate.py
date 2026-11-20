"""Extreme Climate and Non-Stationary EVT Statistics Engine.
Provides advanced non-stationary and threshold extreme value statistics:

1. Non-Stationary Generalized Extreme Value (GEV) Distribution:
   - Time-dependent location parameter: mu(t) = mu_0 + mu_1 * (t - t_0) / 10.0 (decadal trend).
   - Maximum Likelihood Estimation (MLE) with Likelihood Ratio Test (LRT) against stationary baseline.
   - Projected time-varying return levels for future climate horizons (2030, 2050).

2. Peaks-Over-Threshold (POT) Generalized Pareto Distribution (GPD):
   - Extreme event run-length declustering (Coles, 2001) for independent convective storm extraction.
   - GPD parameter estimation and N-year return level extrapolation.
   - Mean Residual Life (MRL) threshold stability diagnostics.

3. Multi-Decadal Latent Regime Causal Filtering Stability Audit:
   - Validates DroughtCascade Gaussian HMM causal filtering stability and normalization.

References:
  - Coles, S. (2001). An Introduction to Statistical Modeling of Extreme Values. Springer.
  - Katz, R. W. (2013). Statistical methods for nonstationary extreme values in climate change.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
from scipy.optimize import minimize

from src.climacore.models.drought.drought_cascade_v2 import DroughtCascadeV2


class NonStationaryGEV:
    """Non-Stationary Generalized Extreme Value (GEV) with time-varying location parameter."""

    def __init__(self):
        self.mu_0: float = 0.0
        self.mu_1: float = 0.0  # Decadal trend
        self.sigma: float = 1.0
        self.xi: float = 0.0    # EVT shape (xi = -c in scipy)
        self.base_year: float = 2000.0
        self.fitted: bool = False
        self.stat_log_lik: float = -float("inf")
        self.non_stat_log_lik: float = -float("inf")
        self.lrt_p_value: float = 1.0

    def fit(self, years: np.ndarray, values: np.ndarray) -> Dict[str, Any]:
        """Fits non-stationary GEV via maximum likelihood with decadal trend mu(t) = mu_0 + mu_1 * (t - t_0)/10."""
        years = np.asarray(years, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        valid = (~np.isnan(years)) & (~np.isnan(values))
        yr = years[valid]
        y = values[valid]

        if len(y) < 15:
            raise ValueError(f"Insufficient annual maxima for non-stationary GEV fit: {len(y)} < 15.")

        self.base_year = float(np.min(yr))
        t_norm = (yr - self.base_year) / 10.0  # Decades from base year

        # Step 1: Fit stationary baseline for LRT comparison
        c_stat, loc_stat, scale_stat = stats.genextreme.fit(y)
        self.stat_log_lik = float(np.sum(stats.genextreme.logpdf(y, c_stat, loc=loc_stat, scale=scale_stat)))

        # Step 2: Fit non-stationary GEV with parameters [mu_0, mu_1, log_sigma, xi]
        init_guess = np.array([loc_stat, 0.1, np.log(max(scale_stat, 1e-3)), -c_stat], dtype=np.float64)

        def neg_log_lik(params):
            mu0, mu1, log_sig, xi_val = params
            sig = np.exp(log_sig)
            if sig <= 1e-4:
                return 1e9

            mu_t = mu0 + mu1 * t_norm
            z = (y - mu_t) / sig

            # Support condition: 1 + xi * z > 0
            if abs(xi_val) > 1e-4:
                cond = 1.0 + xi_val * z
                if np.any(cond <= 0.0):
                    return 1e9
                nll = len(y) * np.log(sig) + (1.0 / xi_val + 1.0) * np.sum(np.log(cond)) + np.sum(cond ** (-1.0 / xi_val))
            else:
                # Gumbel limit
                nll = len(y) * np.log(sig) + np.sum(z) + np.sum(np.exp(-z))

            return float(nll) if np.isfinite(nll) else 1e9

        bounds = [
            (None, None),          # mu_0
            (-10.0, 10.0),         # mu_1 (decadal trend)
            (-3.0, 5.0),           # log_sigma
            (-0.5, 0.5),           # xi (EVT shape)
        ]

        res = minimize(neg_log_lik, init_guess, method="L-BFGS-B", bounds=bounds)
        if res.success or np.isfinite(res.fun):
            self.mu_0 = float(res.x[0])
            self.mu_1 = float(res.x[1])
            self.sigma = float(np.exp(res.x[2]))
            self.xi = float(res.x[3])
            self.non_stat_log_lik = -float(res.fun)
            self.fitted = True

            # Likelihood Ratio Test (LRT): 2 * (lnL_nonstat - lnL_stat) ~ chi2(1)
            lrt_stat = max(0.0, 2.0 * (self.non_stat_log_lik - self.stat_log_lik))
            self.lrt_p_value = float(1.0 - stats.chi2.cdf(lrt_stat, df=1))
        else:
            # Fallback to stationary
            self.mu_0 = loc_stat
            self.mu_1 = 0.0
            self.sigma = scale_stat
            self.xi = -c_stat
            self.non_stat_log_lik = self.stat_log_lik
            self.lrt_p_value = 1.0
            self.fitted = True

        return self.get_summary()

    def predict_return_level(self, return_period_years: float, target_year: float) -> float:
        """Computes time-dependent return level z_T(target_year)."""
        t_norm = (float(target_year) - self.base_year) / 10.0
        mu_t = self.mu_0 + self.mu_1 * t_norm
        p = 1.0 - 1.0 / float(return_period_years)

        if abs(self.xi) > 1e-4:
            y_p = -np.log(p)
            z_p = mu_t + (self.sigma / self.xi) * ((y_p ** (-self.xi)) - 1.0)
        else:
            z_p = mu_t - self.sigma * np.log(-np.log(p))

        return float(z_p)

    def get_summary(self) -> Dict[str, Any]:
        """Summarizes fitted parameters, trend significance, and future return levels."""
        return {
            "model": "Non-Stationary GEV (Linear Location Drift)",
            "base_year": self.base_year,
            "parameters": {
                "mu_0": round(self.mu_0, 3),
                "decadal_trend_mu1": round(self.mu_1, 4),
                "sigma": round(self.sigma, 3),
                "xi_shape": round(self.xi, 4),
            },
            "trend_significance": {
                "stat_log_lik": round(self.stat_log_lik, 3),
                "non_stat_log_lik": round(self.non_stat_log_lik, 3),
                "lrt_p_value": round(self.lrt_p_value, 5),
                "is_trend_significant_p05": bool(self.lrt_p_value < 0.05),
            },
            "projected_return_levels": {
                "T50yr_2030": round(self.predict_return_level(50, 2030), 2),
                "T50yr_2050": round(self.predict_return_level(50, 2050), 2),
                "T100yr_2030": round(self.predict_return_level(100, 2030), 2),
                "T100yr_2050": round(self.predict_return_level(100, 2050), 2),
            },
        }


class PeaksOverThresholdGPD:
    """Peaks-Over-Threshold (POT) Generalized Pareto Distribution with Event Declustering."""

    def __init__(self, run_length_days: int = 3):
        self.run_length_days = int(run_length_days)
        self.threshold: float = 0.0
        self.sigma: float = 1.0
        self.xi: float = 0.0
        self.annual_arrival_rate_lambda: float = 1.0
        self.fitted: bool = False

    def decluster_events(self, values: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts independent peak exceedances separated by at least run_length_days below threshold."""
        v = np.asarray(values, dtype=np.float64)
        th = float(threshold)
        n = len(v)

        cluster_peaks = []
        cluster_indices = []

        in_cluster = False
        curr_peak = -float("inf")
        curr_peak_idx = -1
        days_below = 0

        for t in range(n):
            val = v[t]
            if val > th:
                if not in_cluster:
                    in_cluster = True
                    curr_peak = val
                    curr_peak_idx = t
                else:
                    if val > curr_peak:
                        curr_peak = val
                        curr_peak_idx = t
                days_below = 0
            else:
                if in_cluster:
                    days_below += 1
                    if days_below >= self.run_length_days:
                        cluster_peaks.append(curr_peak)
                        cluster_indices.append(curr_peak_idx)
                        in_cluster = False
                        curr_peak = -float("inf")
                        curr_peak_idx = -1
                        days_below = 0

        # Close final cluster if active
        if in_cluster and curr_peak > th:
            cluster_peaks.append(curr_peak)
            cluster_indices.append(curr_peak_idx)

        return np.array(cluster_peaks, dtype=np.float64), np.array(cluster_indices, dtype=np.int64)

    def fit(self, values: np.ndarray, threshold: float, total_years: float = 30.0) -> Dict[str, Any]:
        """Fits GPD to declustered peak exceedances above threshold."""
        self.threshold = float(threshold)
        peaks, _ = self.decluster_events(values, self.threshold)
        exceedances = peaks - self.threshold

        if len(exceedances) < 5:
            raise ValueError(f"Insufficient declustered peaks above threshold {threshold}: {len(exceedances)} < 5.")

        self.annual_arrival_rate_lambda = float(len(peaks) / max(1.0, float(total_years)))

        # Fit Generalized Pareto Distribution: Scipy genpareto has shape c = xi
        c, loc, scale = stats.genpareto.fit(exceedances, floc=0.0)
        self.xi = float(c)
        self.sigma = float(scale)
        self.fitted = True

        return {
            "threshold": self.threshold,
            "declustered_peaks_count": len(peaks),
            "annual_arrival_rate_lambda": round(self.annual_arrival_rate_lambda, 3),
            "gpd_sigma": round(self.sigma, 4),
            "gpd_xi_shape": round(self.xi, 4),
            "return_levels": {
                "T10yr": round(self.predict_return_level(10), 2),
                "T50yr": round(self.predict_return_level(50), 2),
                "T100yr": round(self.predict_return_level(100), 2),
            },
        }

    def predict_return_level(self, return_period_years: float) -> float:
        """Computes N-year POT return level: x_T = u + (sigma/xi) * [ (lambda * T)^xi - 1 ]."""
        m = self.annual_arrival_rate_lambda * float(return_period_years)
        if abs(self.xi) > 1e-4:
            x_t = self.threshold + (self.sigma / self.xi) * ((m ** self.xi) - 1.0)
        else:
            x_t = self.threshold + self.sigma * np.log(m)
        return float(x_t)


def audit_drought_cascade_forward_stability(n_steps: int = 1000) -> Dict[str, Any]:
    """Audits DroughtCascade HMM forward filtering to verify exact probability conservation and underflow immunity."""
    np.random.seed(42)
    # Instantiate DroughtCascade with synthetic state transitions
    model = DroughtCascadeV2(n_components=4)
    # Synthetic observation matrix mimicking 4 soil variables
    X_synthetic = np.random.normal(0.0, 1.0, (n_steps, 4))
    model.fit(X_synthetic)

    # Compute causal forward filter probabilities
    forward_probs = model.forward_filter_probabilities(X_synthetic)

    # Verifications:
    # 1. Non-negativity
    is_non_neg = bool(np.all(forward_probs >= 0.0))
    # 2. Probability sum == 1.0 at every time step
    row_sums = np.sum(forward_probs, axis=1)
    is_normalized = bool(np.allclose(row_sums, 1.0, atol=1e-6))
    # 3. No NaNs or infinities
    is_finite = bool(np.all(np.isfinite(forward_probs)))

    return {
        "audit_name": "DroughtCascade V2 Causal Forward Filtering Stability Audit",
        "tested_steps": n_steps,
        "is_non_negative": is_non_neg,
        "is_exact_sum_to_one": is_normalized,
        "is_finite_no_underflow": is_finite,
        "min_probability": float(np.min(forward_probs)),
        "max_probability": float(np.max(forward_probs)),
        "status": "PASSED" if (is_non_neg and is_normalized and is_finite) else "FAILED",
    }
