"""Shared Adaptive Uncertainty Calibration Engine for Clima-Core.
Provides theoretically grounded uncertainty quantification and calibration tools:

1. Adaptive Conformal Inference (ACI) (Gibbs & Candès, 2021):
   Dynamically adjusts coverage bounds online under arbitrary distribution drift:
     alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
   where err_t = 1 if y_t not in C_t else 0.

2. Rolling-Window Split Conformal:
   Finite-sample exact marginal coverage guarantees using empirical nonconformity
   scores over a rolling temporal calibration memory.

3. Classification Calibration & Reliability:
   - Platt Scaling (logistic sigmoid calibration on logits/scores)
   - Isotonic Regression (non-parametric monotonic step calibration)
   - Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
   - Reliability diagram binning and curve diagnostics.

References:
  - Gibbs, I., & Candès, E. (2021). Adaptive conformal inference under distribution shift.
    Advances in Neural Information Processing Systems (NeurIPS 34), 1660-1672.
  - Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World.
  - Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern
    neural networks. ICML 2017.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


class AdaptiveConformalInference:
    """Adaptive Conformal Inference (ACI) for dynamic, distribution-shift-robust prediction intervals."""

    def __init__(
        self,
        target_alpha: float = 0.10,
        gamma: float = 0.02,
        initial_quantile: float = 2.0,
        min_alpha: float = 0.01,
        max_alpha: float = 0.99,
        window_size: int = 120,
    ):
        """Initializes ACI.
        
        Parameters:
          target_alpha: Desired miscoverage rate (e.g. 0.10 for 90% coverage, 0.20 for 80% coverage).
          gamma: Online adaptation step size (learning rate, typically 0.005 - 0.05).
          initial_quantile: Initial half-width or nonconformity threshold.
          min_alpha: Lower bound clipping for adaptive alpha.
          max_alpha: Upper bound clipping for adaptive alpha.
          window_size: Rolling window size of nonconformity scores for empirical quantile lookup.
        """
        self.target_alpha = float(target_alpha)
        self.gamma = float(gamma)
        self.curr_alpha = float(target_alpha)
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.window_size = int(window_size)
        self.scores_buffer: List[float] = [float(initial_quantile)]
        self.history: List[Dict[str, float]] = []

    def get_interval(self, y_hat: float) -> Tuple[float, float]:
        """Computes symmetric prediction interval [lower, upper] around point forecast y_hat."""
        # Calculate current empirical quantile based on adapted alpha
        eff_level = 1.0 - self.curr_alpha
        eff_level = max(0.01, min(0.99, eff_level))
        
        if len(self.scores_buffer) == 0:
            q = 1.0
        else:
            # Empirical conformal quantile with finite-sample correction
            n = len(self.scores_buffer)
            idx = int(np.ceil((n + 1) * eff_level)) - 1
            idx = max(0, min(n - 1, idx))
            sorted_scores = sorted(self.scores_buffer)
            q = sorted_scores[idx]
            
        return float(y_hat - q), float(y_hat + q)

    def update(self, y_obs: float, y_hat: float) -> Dict[str, Any]:
        """Observes ground truth y_obs, updates online alpha_t, and records error."""
        lower, upper = self.get_interval(y_hat)
        is_covered = bool(lower <= y_obs <= upper)
        err_t = 0.0 if is_covered else 1.0

        # Nonconformity score for current observation
        score_t = float(abs(y_obs - y_hat))
        self.scores_buffer.append(score_t)
        if len(self.scores_buffer) > self.window_size:
            self.scores_buffer.pop(0)

        # ACI gradient update: alpha_{t+1} = alpha_t + gamma * (target_alpha - err_t)
        prev_alpha = self.curr_alpha
        self.curr_alpha = float(np.clip(
            self.curr_alpha + self.gamma * (self.target_alpha - err_t),
            self.min_alpha,
            self.max_alpha,
        ))

        step_record = {
            "y_hat": round(float(y_hat), 4),
            "y_obs": round(float(y_obs), 4),
            "lower": round(lower, 4),
            "upper": round(upper, 4),
            "is_covered": is_covered,
            "error": err_t,
            "alpha": round(prev_alpha, 4),
            "next_alpha": round(self.curr_alpha, 4),
            "score": round(score_t, 4),
        }
        self.history.append(step_record)
        return step_record

    def get_summary(self) -> Dict[str, Any]:
        """Summarizes online tracking performance and cumulative empirical coverage."""
        if not self.history:
            return {"total_steps": 0, "empirical_coverage_pct": 0.0}
        n = len(self.history)
        covered_count = sum(1 for h in self.history if h["is_covered"])
        empirical_cov = (covered_count / n) * 100.0
        target_cov = (1.0 - self.target_alpha) * 100.0
        widths = [h["upper"] - h["lower"] for h in self.history]
        return {
            "total_steps": n,
            "target_coverage_pct": round(target_cov, 2),
            "empirical_coverage_pct": round(empirical_cov, 2),
            "coverage_gap_pct": round(empirical_cov - target_cov, 2),
            "mean_interval_width": round(float(np.mean(widths)), 4),
            "median_interval_width": round(float(np.median(widths)), 4),
            "final_adapted_alpha": round(self.curr_alpha, 4),
        }


class RollingSplitConformal:
    """Rolling-Window Split Conformal Regressor providing finite-sample valid prediction corridors."""

    def __init__(self, target_coverage: float = 0.80, window_size: int = 180):
        self.target_coverage = float(target_coverage)
        self.alpha = 1.0 - self.target_coverage
        self.window_size = int(window_size)
        self.residuals: List[float] = []

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Initializes rolling residual memory using historical calibration data."""
        res = np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64))
        valid_res = res[~np.isnan(res)]
        self.residuals = list(valid_res[-self.window_size:])

    def get_quantile(self) -> float:
        """Computes the conformal nonconformity quantile at level ceil((n+1)*(1-alpha))/n."""
        if len(self.residuals) == 0:
            return 1.0
        n = len(self.residuals)
        level = (1.0 - self.alpha) * (1.0 + 1.0 / n)
        level = min(1.0, max(0.01, level))
        return float(np.quantile(self.residuals, min(1.0, level)))

    def predict_intervals(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (lower_bounds, upper_bounds) for an array of point forecasts."""
        q = self.get_quantile()
        y_arr = np.asarray(y_pred, dtype=np.float64)
        return y_arr - q, y_arr + q

    def update(self, y_obs: float, y_pred: float):
        """Appends a new residual to the rolling window."""
        res = abs(float(y_obs) - float(y_pred))
        self.residuals.append(res)
        if len(self.residuals) > self.window_size:
            self.residuals.pop(0)


class ClassificationCalibrator:
    """Supervised Classification Probability Calibrator supporting Platt Scaling and Isotonic Regression."""

    def __init__(self, method: str = "isotonic"):
        """Initializes calibrator with method 'isotonic' or 'platt'."""
        if method not in ("isotonic", "platt"):
            raise ValueError(f"Unknown calibration method: {method}. Must be 'isotonic' or 'platt'.")
        self.method = method
        self.model: Optional[Union[IsotonicRegression, LogisticRegression]] = None
        self.fitted = False

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray):
        """Fits calibration mapping from raw predicted probabilities to empirical outcomes."""
        p = np.asarray(y_prob, dtype=np.float64)
        y = np.asarray(y_true, dtype=np.float64)
        valid = (~np.isnan(p)) & (~np.isnan(y))
        p_clean = p[valid]
        y_clean = y[valid]

        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.model.fit(p_clean, y_clean)
        else:  # platt
            # Log-odds / logit transform with clipping
            p_clipped = np.clip(p_clean, 1e-5, 1.0 - 1e-5)
            logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
            self.model = LogisticRegression(C=1.0, solver="lbfgs")
            self.model.fit(logits, y_clean)

        self.fitted = True

    def predict_proba(self, y_prob: np.ndarray) -> np.ndarray:
        """Transforms raw probabilities to calibrated probabilities strictly in [0, 1]."""
        if not self.fitted or self.model is None:
            return np.asarray(y_prob, dtype=np.float64)
        p = np.asarray(y_prob, dtype=np.float64)
        if self.method == "isotonic":
            cal_p = self.model.predict(p)
            return np.clip(cal_p, 0.0, 1.0)
        else:  # platt
            p_clipped = np.clip(p, 1e-5, 1.0 - 1e-5)
            logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
            cal_p = self.model.predict_proba(logits)[:, 1]
            return np.clip(cal_p, 0.0, 1.0)

    @staticmethod
    def compute_calibration_metrics(
        y_prob: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        """Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier Score.
        
        ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
        """
        p = np.asarray(y_prob, dtype=np.float64)
        y = np.asarray(y_true, dtype=np.float64)
        valid = (~np.isnan(p)) & (~np.isnan(y))
        p = p[valid]
        y = y[valid]

        n = len(p)
        if n == 0:
            return {"ece": 0.0, "mce": 0.0, "brier_score": 0.0, "bins": []}

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        mce = 0.0
        bins_data: List[Dict[str, Any]] = []

        for i in range(n_bins):
            b_min, b_max = bin_edges[i], bin_edges[i + 1]
            if i == n_bins - 1:
                mask = (p >= b_min) & (p <= b_max)
            else:
                mask = (p >= b_min) & (p < b_max)

            count = int(np.sum(mask))
            if count > 0:
                bin_acc = float(np.mean(y[mask]))
                bin_conf = float(np.mean(p[mask]))
                diff = abs(bin_acc - bin_conf)
                ece += (count / n) * diff
                mce = max(mce, diff)
                bins_data.append({
                    "bin_index": i,
                    "bin_range": [round(b_min, 2), round(b_max, 2)],
                    "count": count,
                    "mean_confidence": round(bin_conf, 4),
                    "empirical_accuracy": round(bin_acc, 4),
                    "absolute_calibration_error": round(diff, 4),
                })
            else:
                bins_data.append({
                    "bin_index": i,
                    "bin_range": [round(b_min, 2), round(b_max, 2)],
                    "count": 0,
                    "mean_confidence": round((b_min + b_max) / 2.0, 2),
                    "empirical_accuracy": 0.0,
                    "absolute_calibration_error": 0.0,
                })

        brier = float(np.mean((p - y) ** 2))

        return {
            "n_samples": n,
            "ece": round(float(ece), 4),
            "mce": round(float(mce), 4),
            "brier_score": round(brier, 4),
            "reliability_diagram_bins": bins_data,
        }
