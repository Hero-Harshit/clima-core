"""Advanced probabilistic ensemble architectures for ForecastFusion V1.
Includes:
- Model A: Weighted Forecast Ensemble (constrained SLSQP optimization)
- Model B: EMOS (Ensemble Model Output Statistics with CRPS loss minimization)
- Model C: NGBoost Distributional Regressor (Normal / LogNormal distributions)
- Model D: Quantile LightGBM with Two-Part Hurdle Model for Precipitation
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import scipy.optimize as opt
from scipy.stats import norm
import lightgbm as lgb
from ngboost import NGBRegressor
from ngboost.distns import Normal, LogNormal
from ngboost.scores import LogScore
import joblib


# -------------------------------------------------------------------------
# Helper Functions: CRPS for Gaussian Distributions
# -------------------------------------------------------------------------

def gaussian_crps(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes exact closed-form CRPS for Gaussian forecast distribution N(mu, sigma^2)."""
    sigma = np.maximum(sigma, 1e-4)
    z = (y - mu) / sigma
    pdf_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    crps = sigma * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - 1.0 / np.sqrt(np.pi))
    return crps


# -------------------------------------------------------------------------
# Model A: Weighted Forecast Ensemble
# -------------------------------------------------------------------------

class WeightedForecastEnsemble:
    """Non-negative constrained ensemble with bias intercept: y_hat = a + sum(w_k * x_k)."""
    
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.weights_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.feature_names_: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "WeightedForecastEnsemble":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_features = X.shape[1]
        self.feature_names_ = feature_names or [f"model_{i}" for i in range(n_features)]
        
        # Initial guess: equal weights, zero intercept
        if self.fit_intercept:
            x0 = np.concatenate([[0.0], np.ones(n_features) / n_features])
            bounds = [(None, None)] + [(0.0, 1.0) for _ in range(n_features)]
            
            def loss(params):
                a = params[0]
                w = params[1:]
                pred = a + X @ w
                return np.mean((y - pred) ** 2)
            
            constraints = [{"type": "eq", "fun": lambda params: np.sum(params[1:]) - 1.0}]
            res = opt.minimize(loss, x0, method="SLSQP", bounds=bounds, constraints=constraints)
            self.intercept_ = float(res.x[0])
            self.weights_ = res.x[1:]
        else:
            x0 = np.ones(n_features) / n_features
            bounds = [(0.0, 1.0) for _ in range(n_features)]
            
            def loss(w):
                pred = X @ w
                return np.mean((y - pred) ** 2)
                
            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
            res = opt.minimize(loss, x0, method="SLSQP", bounds=bounds, constraints=constraints)
            self.intercept_ = 0.0
            self.weights_ = res.x
            
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return self.intercept_ + X @ self.weights_

    def get_weights_dict(self) -> Dict[str, float]:
        res = {"intercept": round(self.intercept_, 4)}
        for name, w in zip(self.feature_names_, self.weights_):
            res[name] = round(float(w), 4)
        return res


# -------------------------------------------------------------------------
# Model B: EMOS (Ensemble Model Output Statistics)
# -------------------------------------------------------------------------

class EMOSModel:
    """Ensemble Model Output Statistics (Gneiting et al. 2005).
    Predicts Gaussian distribution N(mu, sigma^2) where:
      mu = a + sum(b_k * x_k),  b_k >= 0
      sigma^2 = c + d * s^2,    c > 0, d >= 0
    where s^2 is the sample variance across the input ensemble forecasts.
    Optimized directly via CRPS loss.
    """
    
    def __init__(self):
        self.params_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "EMOSModel":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_models = X.shape
        self.feature_names_ = feature_names or [f"model_{i}" for i in range(n_models)]
        
        # Ensemble variance per sample
        ens_var = np.var(X, axis=1, ddof=1) if n_models > 1 else np.ones(n_samples)
        
        # Parameter vector: [a, b_1, ..., b_K, c, d]
        # Length: 1 + K + 2 = K + 3
        # Bounds: a (unconstrained), b_k >= 0, c >= 1e-4, d >= 0
        x0 = np.concatenate([[0.0], np.ones(n_models) / n_models, [1.0, 1.0]])
        bounds = [(None, None)] + [(0.0, None) for _ in range(n_models)] + [(1e-4, None), (0.0, None)]
        
        def crps_objective(params):
            a = params[0]
            b = params[1:1+n_models]
            c = params[1+n_models]
            d = params[2+n_models]
            
            mu = a + X @ b
            sigma2 = np.maximum(c + d * ens_var, 1e-4)
            sigma = np.sqrt(sigma2)
            
            val = np.mean(gaussian_crps(mu, sigma, y))
            return val
            
        res = opt.minimize(crps_objective, x0, method="L-BFGS-B", bounds=bounds)
        self.params_ = res.x
        return self

    def predict_distribution(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (mu, sigma) arrays for test samples."""
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_models = X.shape
        ens_var = np.var(X, axis=1, ddof=1) if n_models > 1 else np.ones(n_samples)
        
        a = self.params_[0]
        b = self.params_[1:1+n_models]
        c = self.params_[1+n_models]
        d = self.params_[2+n_models]
        
        mu = a + X @ b
        sigma = np.sqrt(np.maximum(c + d * ens_var, 1e-4))
        return mu, sigma

    def predict(self, X: np.ndarray) -> np.ndarray:
        mu, _ = self.predict_distribution(X)
        return mu

    def get_params_dict(self) -> Dict[str, float]:
        n_models = len(self.feature_names_)
        return {
            "a_intercept": round(float(self.params_[0]), 4),
            "weights": {name: round(float(self.params_[1+i]), 4) for i, name in enumerate(self.feature_names_)},
            "c_base_variance": round(float(self.params_[1+n_models]), 4),
            "d_spread_coef": round(float(self.params_[2+n_models]), 4),
        }


# -------------------------------------------------------------------------
# Model C: NGBoost Distributional Regressor
# -------------------------------------------------------------------------

class NGBoostForecastFusion:
    """NGBoost probabilistic gradient boosting regressor for temperature or rainfall."""
    
    def __init__(self, n_estimators: int = 150, learning_rate: float = 0.03, dist: str = "normal"):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.dist_name = dist
        self.dist_cls = Normal if dist == "normal" else LogNormal
        self.model: Optional[NGBRegressor] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NGBoostForecastFusion":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        if self.dist_name == "lognormal":
            y = np.maximum(y, 1e-3)
            
        self.model = NGBRegressor(
            Dist=self.dist_cls,
            Score=LogScore,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            verbose=False,
            random_state=42,
        )
        self.model.fit(X, y)
        return self

    def predict_distribution(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        dist = self.model.pred_dist(X)
        if self.dist_name == "normal":
            mu = dist.loc
            sigma = dist.scale
        else:
            # LogNormal returns s (scale) and scale (exp(loc))
            mu = np.exp(dist.loc + 0.5 * dist.scale ** 2)
            sigma = np.sqrt((np.exp(dist.scale ** 2) - 1.0) * np.exp(2 * dist.loc + dist.scale ** 2))
        return mu, sigma

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return self.model.predict(X)


# -------------------------------------------------------------------------
# Model D: Quantile LightGBM and Two-Part Precipitation Ensemble
# -------------------------------------------------------------------------

class QuantileLGBMEnsemble:
    """Multi-quantile LightGBM regressor predicting Q10, Q50, Q90."""
    
    def __init__(self, quantiles: List[float] = [0.10, 0.50, 0.90], n_estimators: int = 120, max_depth: int = 4):
        self.quantiles = quantiles
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models_: Dict[float, lgb.LGBMRegressor] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantileLGBMEnsemble":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        for q in self.quantiles:
            reg = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                num_leaves=15,
                learning_rate=0.04,
                n_jobs=2,
                random_state=42,
                verbose=-1,
            )
            reg.fit(X, y)
            self.models_[q] = reg
        return self

    def predict_quantiles(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        X = np.asarray(X, dtype=np.float32)
        preds = {q: self.models_[q].predict(X) for q in self.quantiles}
        # Enforce quantile monotonicity: Q10 <= Q50 <= Q90
        q10 = preds[0.10]
        q50 = np.maximum(preds[0.50], q10)
        q90 = np.maximum(preds[0.90], q50)
        return {0.10: q10, 0.50: q50, 0.90: q90}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_quantiles(X)[0.50]


class TwoPartPrecipitationEnsemble:
    """Two-part Hurdle Ensemble for zero-inflated precipitation:
    Part 1: Classifier for Rain Occurrence Probability P(Rain > threshold)
    Part 2: Quantile Regressors for conditional positive rainfall amounts given rain.
    """
    
    def __init__(self, threshold: float = 0.1, n_estimators: int = 100):
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.04,
            n_jobs=2,
            random_state=42,
            verbose=-1,
        )
        self.amount_reg = QuantileLGBMEnsemble(quantiles=[0.10, 0.50, 0.90], n_estimators=n_estimators, max_depth=4)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TwoPartPrecipitationEnsemble":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        # Train Part 1: Rain classification
        is_rain = (y > self.threshold).astype(int)
        self.clf.fit(X, is_rain)
        
        # Train Part 2: Amount distribution conditional on positive rain
        rain_mask = is_rain == 1
        if np.sum(rain_mask) > 10:
            self.amount_reg.fit(X[rain_mask], y[rain_mask])
        else:
            self.amount_reg.fit(X, y)
            
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability of precipitation P(Rain > threshold)."""
        X = np.asarray(X, dtype=np.float32)
        return self.clf.predict_proba(X)[:, 1]

    def predict_expected(self, X: np.ndarray) -> np.ndarray:
        """Returns expected rainfall amount: E[Y] = P(Rain) * E[Y | Rain]."""
        p_rain = self.predict_proba(X)
        cond_median = self.amount_reg.predict(X)
        return p_rain * np.maximum(cond_median, 0.0)

    def predict_distribution(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        p_rain = self.predict_proba(X)
        amounts = self.amount_reg.predict_quantiles(X)
        return {
            "p_rain": p_rain,
            "q10_cond": amounts[0.10],
            "q50_cond": amounts[0.50],
            "q90_cond": amounts[0.90],
            "expected_amount": p_rain * np.maximum(amounts[0.50], 0.0),
        }
