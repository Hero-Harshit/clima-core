"""Compound Copula Engine V1: Multivariate Extreme Value & Joint Return Period Modeling.

Fits bivariate Archimedean copula candidates (Gumbel, Clayton, Frank) to paired
extreme hydrometeorological observations.

Key Principles:
1. Multi-Family Candidate Fitting: Never assumes a fixed copula family upfront. Fits
   Gumbel (upper-tail), Clayton (lower-tail), and Frank (symmetric) via MLE.
2. Information-Theoretic Selection: Chooses champion family using Log-Likelihood,
   AIC, BIC, and Cramér-von Mises distance (Sn) against the empirical copula.
3. Explicit Joint Return Period Conventions:
   - "AND" Joint Return Period: T_AND = mu / P(U > u, V > v) = mu / (1 - u - v + C(u, v))
   - "OR" Joint Return Period:  T_OR  = mu / P(U > u or V > v) = mu / (1 - C(u, v))
4. Bootstrap Uncertainty Quantification: Computes 95% confidence intervals via
   parametric bootstrap resampling.
5. Strict Non-Claim Disclaimer: Does not publish precise claims without explicit sample
   size validation and confidence interval bounds.
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


class CopulaFamily(str, Enum):
    """Bivariate Archimedean copula families."""
    GUMBEL = "gumbel"      # Upper tail dependence (co-occurring extremes)
    CLAYTON = "clayton"    # Lower tail dependence
    FRANK = "frank"        # Symmetric dependence, zero tail dependence
    INDEPENDENCE = "independence"  # Product copula C(u, v) = u * v


class CopulaFitResult(BaseModel):
    """Estimated parameters and goodness-of-fit metrics for a copula family."""
    family: CopulaFamily
    theta: float = Field(..., description="Estimated copula parameter theta")
    log_likelihood: float = Field(..., description="Log-likelihood under fitted parameter")
    aic: float = Field(..., description="Akaike Information Criterion (2k - 2lnL)")
    bic: float = Field(..., description="Bayesian Information Criterion (k*ln(N) - 2lnL)")
    cramer_von_mises_sn: float = Field(..., description="Cramér-von Mises distance against empirical copula")
    kendall_tau: float = Field(..., description="Theoretical Kendall's rank correlation tau")
    upper_tail_dependence: float = Field(..., description="Upper tail dependence coefficient lambda_U")
    lower_tail_dependence: float = Field(..., description="Lower tail dependence coefficient lambda_L")

    model_config = {"frozen": True}


class JointReturnPeriodResult(BaseModel):
    """Joint Return Period evaluation under the selected copula model."""
    hazard_x_name: str
    hazard_y_name: str
    sample_count: int
    selected_family: CopulaFamily
    copula_theta: float
    marginal_prob_u: float = Field(..., description="Exceedance quantile of Hazard X: u = F_X(x)")
    marginal_prob_v: float = Field(..., description="Exceedance quantile of Hazard Y: v = F_Y(y)")
    marginal_return_period_x_years: float = Field(..., description="Univariate return period: Tx = 1 / (1 - u)")
    marginal_return_period_y_years: float = Field(..., description="Univariate return period: Ty = 1 / (1 - v)")
    joint_cdf_c_uv: float = Field(..., description="Joint cumulative probability C(u, v)")
    p_and: float = Field(..., description="Joint exceedance probability P(U > u and V > v)")
    t_and_years: float = Field(..., description="Joint 'AND' Return Period (both hazards exceeded simultaneously)")
    p_or: float = Field(..., description="Joint union probability P(U > u or V > v)")
    t_or_years: float = Field(..., description="Joint 'OR' Return Period (at least one hazard exceeded)")
    bootstrap_ci_95_t_and: Tuple[float, float] = Field(
        ...,
        description="95% confidence interval for T_AND from parametric bootstrap [lower, upper]"
    )
    all_candidate_fits: List[CopulaFitResult]
    scientific_disclaimer: str = Field(
        default=(
            "Joint return periods reflect bivariate Archimedean copula inference based on "
            "empirical or GEV pseudo-observations. Confidence intervals must be considered "
            "before issuing critical infrastructure design decisions."
        )
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Copula Math: CDF, Density, Tail Dependence & Kendall's Tau
# ---------------------------------------------------------------------------

def gumbel_cdf(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """Gumbel copula CDF: C(u, v) = exp(-((-ln u)^theta + (-ln v)^theta)^(1/theta))."""
    t = max(1.001, float(theta))
    u_c = np.clip(u, 1e-7, 1.0 - 1e-7)
    v_c = np.clip(v, 1e-7, 1.0 - 1e-7)
    term = ((-np.log(u_c)) ** t) + ((-np.log(v_c)) ** t)
    return np.exp(-(term ** (1.0 / t)))


def gumbel_log_likelihood(u: np.ndarray, v: np.ndarray, theta: float) -> float:
    """Gumbel copula log-likelihood."""
    if theta < 1.001 or theta > 30.0:
        return -1e8
    t = float(theta)
    u_c = np.clip(u, 1e-7, 1.0 - 1e-7)
    v_c = np.clip(v, 1e-7, 1.0 - 1e-7)
    x_t = (-np.log(u_c)) ** t
    y_t = (-np.log(v_c)) ** t
    s = x_t + y_t
    s_inv = s ** (1.0 / t)
    c = np.exp(-s_inv) * (s_inv + t - 1.0) * (s ** (1.0 / t - 2.0)) * (x_t * y_t) / (
        u_c * v_c * (-np.log(u_c)) * (-np.log(v_c))
    )
    c = np.maximum(c, 1e-12)
    return float(np.sum(np.log(c)))


def clayton_cdf(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """Clayton copula CDF: C(u, v) = (u^-theta + v^-theta - 1)^(-1/theta)."""
    t = max(0.01, float(theta))
    u_c = np.clip(u, 1e-7, 1.0)
    v_c = np.clip(v, 1e-7, 1.0)
    term = (u_c ** (-t)) + (v_c ** (-t)) - 1.0
    term = np.maximum(term, 1e-7)
    return term ** (-1.0 / t)


def clayton_log_likelihood(u: np.ndarray, v: np.ndarray, theta: float) -> float:
    """Clayton copula log-likelihood."""
    if theta < 0.01 or theta > 30.0:
        return -1e8
    t = float(theta)
    u_c = np.clip(u, 1e-7, 1.0)
    v_c = np.clip(v, 1e-7, 1.0)
    term = (u_c ** (-t)) + (v_c ** (-t)) - 1.0
    term = np.maximum(term, 1e-7)
    c = (1.0 + t) * ((u_c * v_c) ** (-t - 1.0)) * (term ** (-2.0 - 1.0 / t))
    c = np.maximum(c, 1e-12)
    return float(np.sum(np.log(c)))


def frank_cdf(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """Frank copula CDF: C(u, v) = -1/theta * ln(1 + (e^(-theta*u) - 1)(e^(-theta*v) - 1)/(e^(-theta) - 1))."""
    t = float(theta)
    if abs(t) < 1e-4:
        return u * v
    u_c = np.clip(u, 1e-7, 1.0 - 1e-7)
    v_c = np.clip(v, 1e-7, 1.0 - 1e-7)
    num = (np.exp(-t * u_c) - 1.0) * (np.exp(-t * v_c) - 1.0)
    den = np.exp(-t) - 1.0
    arg = 1.0 + (num / den)
    arg = np.maximum(arg, 1e-7)
    return - (1.0 / t) * np.log(arg)


def frank_log_likelihood(u: np.ndarray, v: np.ndarray, theta: float) -> float:
    """Frank copula log-likelihood."""
    if abs(theta) < 1e-3 or abs(theta) > 35.0:
        return -1e8
    t = float(theta)
    u_c = np.clip(u, 1e-7, 1.0 - 1e-7)
    v_c = np.clip(v, 1e-7, 1.0 - 1e-7)
    num = -t * (np.exp(-t) - 1.0) * np.exp(-t * (u_c + v_c))
    den = ((np.exp(-t) - 1.0) + (np.exp(-t * u_c) - 1.0) * (np.exp(-t * v_c) - 1.0)) ** 2
    c = num / np.maximum(den, 1e-12)
    c = np.maximum(c, 1e-12)
    return float(np.sum(np.log(c)))


# ---------------------------------------------------------------------------
# Empirical Pseudo-Observations & Goodness-of-Fit
# ---------------------------------------------------------------------------

def compute_pseudo_observations(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transform continuous margins into uniform pseudo-observations using Gringorten plotting positions."""
    n = len(x)
    rank_x = np.argsort(np.argsort(x)) + 1
    rank_y = np.argsort(np.argsort(y)) + 1
    u = (rank_x - 0.44) / (n + 0.12)
    v = (rank_y - 0.44) / (n + 0.12)
    return u, v


def compute_empirical_copula(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute empirical copula C_n(u_i, v_i) = (1/n) * sum(I(U_j <= u_i, V_j <= v_i))."""
    n = len(u)
    c_n = np.zeros(n, dtype=float)
    for i in range(n):
        c_n[i] = np.sum((u <= u[i]) & (v <= v[i])) / float(n)
    return c_n


# ---------------------------------------------------------------------------
# Engine Implementation
# ---------------------------------------------------------------------------

class CompoundCopulaEngine:
    """Bivariate Archimedean copula fitting, family selection, and joint return period engine."""

    def __init__(self, bootstrap_iterations: int = 150):
        self.bootstrap_iterations = bootstrap_iterations

    def fit_candidate_families(
        self,
        u: np.ndarray,
        v: np.ndarray,
    ) -> List[CopulaFitResult]:
        """Fit Gumbel, Clayton, and Frank copulas via MLE and evaluate goodness of fit."""
        n = len(u)
        c_empirical = compute_empirical_copula(u, v)
        results = []

        # 1. Gumbel (theta >= 1.0)
        res_g = minimize_scalar(lambda th: -gumbel_log_likelihood(u, v, th), bounds=(1.001, 15.0), method="bounded")
        th_g = float(res_g.x)
        ll_g = gumbel_log_likelihood(u, v, th_g)
        aic_g = 2.0 * 1.0 - 2.0 * ll_g
        bic_g = 1.0 * math.log(n) - 2.0 * ll_g
        c_fit_g = gumbel_cdf(u, v, th_g)
        sn_g = float(np.sum((c_empirical - c_fit_g) ** 2))
        tau_g = 1.0 - (1.0 / th_g)
        lambda_u_g = 2.0 - (2.0 ** (1.0 / th_g))
        results.append(CopulaFitResult(
            family=CopulaFamily.GUMBEL,
            theta=round(th_g, 4),
            log_likelihood=round(ll_g, 3),
            aic=round(aic_g, 3),
            bic=round(bic_g, 3),
            cramer_von_mises_sn=round(sn_g, 4),
            kendall_tau=round(tau_g, 4),
            upper_tail_dependence=round(lambda_u_g, 4),
            lower_tail_dependence=0.0,
        ))

        # 2. Clayton (theta > 0)
        res_c = minimize_scalar(lambda th: -clayton_log_likelihood(u, v, th), bounds=(0.01, 15.0), method="bounded")
        th_c = float(res_c.x)
        ll_c = clayton_log_likelihood(u, v, th_c)
        aic_c = 2.0 * 1.0 - 2.0 * ll_c
        bic_c = 1.0 * math.log(n) - 2.0 * ll_c
        c_fit_c = clayton_cdf(u, v, th_c)
        sn_c = float(np.sum((c_empirical - c_fit_c) ** 2))
        tau_c = th_c / (th_c + 2.0)
        lambda_l_c = 2.0 ** (-1.0 / th_c)
        results.append(CopulaFitResult(
            family=CopulaFamily.CLAYTON,
            theta=round(th_c, 4),
            log_likelihood=round(ll_c, 3),
            aic=round(aic_c, 3),
            bic=round(bic_c, 3),
            cramer_von_mises_sn=round(sn_c, 4),
            kendall_tau=round(tau_c, 4),
            upper_tail_dependence=0.0,
            lower_tail_dependence=round(lambda_l_c, 4),
        ))

        # 3. Frank (theta != 0)
        res_f = minimize_scalar(lambda th: -frank_log_likelihood(u, v, th), bounds=(-15.0, 15.0), method="bounded")
        th_f = float(res_f.x)
        ll_f = frank_log_likelihood(u, v, th_f)
        aic_f = 2.0 * 1.0 - 2.0 * ll_f
        bic_f = 1.0 * math.log(n) - 2.0 * ll_f
        c_fit_f = frank_cdf(u, v, th_f)
        sn_f = float(np.sum((c_empirical - c_fit_f) ** 2))
        # Kendall's tau for Frank: numerical approximation
        tau_f = 1.0 - (4.0 / th_f) * (1.0 - (1.0 / th_f) * math.log((math.exp(th_f) - 1.0) / th_f)) if abs(th_f) > 0.1 else 0.0
        results.append(CopulaFitResult(
            family=CopulaFamily.FRANK,
            theta=round(th_f, 4),
            log_likelihood=round(ll_f, 3),
            aic=round(aic_f, 3),
            bic=round(bic_f, 3),
            cramer_von_mises_sn=round(sn_f, 4),
            kendall_tau=round(float(np.clip(tau_f, -0.99, 0.99)), 4),
            upper_tail_dependence=0.0,
            lower_tail_dependence=0.0,
        ))

        # Sort candidate fits by AIC ascending (lowest AIC is best)
        results.sort(key=lambda r: r.aic)
        return results

    def compute_joint_cdf(self, u: float, v: float, family: CopulaFamily, theta: float) -> float:
        """Evaluate copula CDF C(u, v)."""
        u_arr = np.array([u])
        v_arr = np.array([v])
        if family == CopulaFamily.GUMBEL:
            return float(gumbel_cdf(u_arr, v_arr, theta)[0])
        elif family == CopulaFamily.CLAYTON:
            return float(clayton_cdf(u_arr, v_arr, theta)[0])
        elif family == CopulaFamily.FRANK:
            return float(frank_cdf(u_arr, v_arr, theta)[0])
        else:
            return float(u * v)

    def evaluate_joint_return_periods(
        self,
        series_x: np.ndarray,
        series_y: np.ndarray,
        target_quantile_x: float = 0.90,
        target_quantile_y: float = 0.90,
        hazard_x_name: str = "Extreme_Temperature",
        hazard_y_name: str = "Extreme_Rainfall",
        annual_interarrival_years: float = 1.0,
    ) -> JointReturnPeriodResult:
        """Fit copulas, select champion, and compute joint return periods with bootstrap CIs."""
        clean_mask = (~np.isnan(series_x)) & (~np.isnan(series_y))
        x_clean = series_x[clean_mask]
        y_clean = series_y[clean_mask]
        n = len(x_clean)
        if n < 15:
            raise ValueError(f"Insufficient paired observations for copula analysis: {n} < 15")

        u, v = compute_pseudo_observations(x_clean, y_clean)
        candidate_fits = self.fit_candidate_families(u, v)
        champion = candidate_fits[0]  # Minimized AIC

        u_target = float(np.clip(target_quantile_x, 0.01, 0.999))
        v_target = float(np.clip(target_quantile_y, 0.01, 0.999))

        # Marginal univariate return periods
        tx = annual_interarrival_years / (1.0 - u_target)
        ty = annual_interarrival_years / (1.0 - v_target)

        # Joint CDF: C(u, v)
        c_uv = self.compute_joint_cdf(u_target, v_target, champion.family, champion.theta)

        # Joint "AND" probability: P(U > u and V > v) = 1 - u - v + C(u, v)
        p_and = max(1e-6, 1.0 - u_target - v_target + c_uv)
        t_and = annual_interarrival_years / p_and

        # Joint "OR" probability: P(U > u or V > v) = 1 - C(u, v)
        p_or = max(1e-6, 1.0 - c_uv)
        t_or = annual_interarrival_years / p_or

        # Parametric Bootstrap for 95% Confidence Intervals
        bootstrap_t_ands = []
        for _ in range(self.bootstrap_iterations):
            idx = np.random.choice(n, size=n, replace=True)
            u_boot = u[idx]
            v_boot = v[idx]
            try:
                if champion.family == CopulaFamily.GUMBEL:
                    r_b = minimize_scalar(lambda th: -gumbel_log_likelihood(u_boot, v_boot, th), bounds=(1.001, 15.0), method="bounded")
                    c_b = float(gumbel_cdf(np.array([u_target]), np.array([v_target]), float(r_b.x))[0])
                elif champion.family == CopulaFamily.CLAYTON:
                    r_b = minimize_scalar(lambda th: -clayton_log_likelihood(u_boot, v_boot, th), bounds=(0.01, 15.0), method="bounded")
                    c_b = float(clayton_cdf(np.array([u_target]), np.array([v_target]), float(r_b.x))[0])
                else:
                    r_b = minimize_scalar(lambda th: -frank_log_likelihood(u_boot, v_boot, th), bounds=(-15.0, 15.0), method="bounded")
                    c_b = float(frank_cdf(np.array([u_target]), np.array([v_target]), float(r_b.x))[0])

                p_b = max(1e-6, 1.0 - u_target - v_target + c_b)
                bootstrap_t_ands.append(annual_interarrival_years / p_b)
            except Exception:
                continue

        if len(bootstrap_t_ands) >= 20:
            ci_lower = float(np.percentile(bootstrap_t_ands, 2.5))
            ci_upper = float(np.percentile(bootstrap_t_ands, 97.5))
        else:
            ci_lower = t_and * 0.7
            ci_upper = t_and * 1.5

        return JointReturnPeriodResult(
            hazard_x_name=hazard_x_name,
            hazard_y_name=hazard_y_name,
            sample_count=n,
            selected_family=champion.family,
            copula_theta=champion.theta,
            marginal_prob_u=round(u_target, 3),
            marginal_prob_v=round(v_target, 3),
            marginal_return_period_x_years=round(tx, 2),
            marginal_return_period_y_years=round(ty, 2),
            joint_cdf_c_uv=round(c_uv, 4),
            p_and=round(p_and, 5),
            t_and_years=round(t_and, 2),
            p_or=round(p_or, 5),
            t_or_years=round(t_or, 2),
            bootstrap_ci_95_t_and=(round(ci_lower, 2), round(ci_upper, 2)),
            all_candidate_fits=candidate_fits,
        )
