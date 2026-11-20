"""Hybrid GR4J Hydrological Model.
Combines the physical mass-conserving 4-parameter GR4J conceptual model with a
regularized statistical/ML residual learner conditioned on antecedent hydro-meteorological
states and internal GR4J reservoir levels.

Formulation:
  Q_hybrid(t) = max(0.0, Q_GR4J(t) + f_theta(X(t)))

Where:
  - Q_GR4J(t): Baseline discharge from calibrated physical GR4J simulation.
  - f_theta: Residual learner (Ridge / LightGBM) predicting unmodelled catchment dynamics:
      e(t) = Q_obs(t) - Q_GR4J(t)
  - X(t): Multi-scale antecedent state vector:
      [Q_GR4J(t), S(t)/x1, R(t)/x3, P_3d, P_7d, P_14d, P_30d, PET_7d, sin(DOY), cos(DOY)]
  - Non-negativity is strictly enforced via the ReLU / max(0, .) boundary condition.

References:
  - Perrin, C., Michel, C., & Andréassian, V. (2003). Improvement of a parsimonious
    model for streamflow simulation. Journal of Hydrology, 279(1-4), 275-289.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.climacore.models.basinflow.conceptual_models import GR4J
from src.climacore.models.basinflow.metrics import compute_hydrological_metrics


class HybridGR4J:
    """Hybrid Conceptual-ML Hydrological Model pairing GR4J with a regularized residual learner."""

    def __init__(
        self,
        gr4j_params: Optional[np.ndarray] = None,
        alpha: float = 25.0,
        use_lgbm: bool = False,
    ):
        """Initializes HybridGR4J.
        
        Parameters:
          gr4j_params: Optional 4-parameter vector [x1, x2, x3, x4] for GR4J.
          alpha: Regularization strength for Ridge residual learner.
          use_lgbm: If True, uses LightGBM instead of Ridge for residual correction.
        """
        if gr4j_params is not None:
            self.gr4j = GR4J(*gr4j_params)
            self.gr4j.calibrated = True
        else:
            self.gr4j = GR4J()
        self.alpha = float(alpha)
        self.use_lgbm = bool(use_lgbm)
        self.scaler = StandardScaler()
        self.residual_model: Optional[Union[Ridge, Any]] = None
        self.calibrated = False
        self.feature_names = [
            "q_gr4j",
            "s_ratio",
            "r_ratio",
            "p_curr",
            "p_3d",
            "p_7d",
            "p_14d",
            "p_30d",
            "pet_curr",
            "pet_7d",
            "sin_doy",
            "cos_doy",
        ]

    def _extract_features(
        self,
        prcp: np.ndarray,
        pet: np.ndarray,
        dates: Optional[Union[pd.DatetimeIndex, List[Any], np.ndarray]] = None,
        gr4j_sim_res: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts multi-scale antecedent meteorological and internal reservoir state features.
        
        Returns:
          (feature_matrix X, base_gr4j_q)
        """
        n = len(prcp)
        if gr4j_sim_res is None:
            gr4j_sim_res = self.gr4j.simulate(prcp, pet, return_stores=True)

        q_gr4j = gr4j_sim_res["q_sim"]
        s_store = gr4j_sim_res.get("s_store", np.zeros(n))
        r_store = gr4j_sim_res.get("r_store", np.zeros(n))

        x1 = max(float(self.gr4j.params[0]), 1.0)
        x3 = max(float(self.gr4j.params[2]), 1.0)

        s_ratio = s_store / x1
        r_ratio = r_store / x3

        # Rolling antecedent precipitation sums
        p_series = pd.Series(prcp, dtype=np.float64)
        p_3d = p_series.rolling(3, min_periods=1).sum().values
        p_7d = p_series.rolling(7, min_periods=1).sum().values
        p_14d = p_series.rolling(14, min_periods=1).sum().values
        p_30d = p_series.rolling(30, min_periods=1).sum().values

        # Rolling antecedent PET mean
        pet_series = pd.Series(pet, dtype=np.float64)
        pet_7d = pet_series.rolling(7, min_periods=1).mean().values

        # Harmonic seasonality
        if dates is not None:
            dt_idx = pd.to_datetime(dates)
            doy = dt_idx.dt.dayofyear.values if hasattr(dt_idx, "dt") else dt_idx.dayofyear.values
        else:
            doy = (np.arange(n) % 365) + 1

        sin_doy = np.sin(2.0 * np.pi * doy / 365.25)
        cos_doy = np.cos(2.0 * np.pi * doy / 365.25)

        features = np.column_stack([
            q_gr4j,
            s_ratio,
            r_ratio,
            prcp,
            p_3d,
            p_7d,
            p_14d,
            p_30d,
            pet,
            pet_7d,
            sin_doy,
            cos_doy,
        ])

        return features, q_gr4j

    def calibrate(
        self,
        prcp: np.ndarray,
        pet: np.ndarray,
        q_obs: np.ndarray,
        dates: Optional[Any] = None,
        warmup_days: int = 365,
    ) -> Dict[str, Any]:
        """Calibrates GR4J physical parameters first, then fits the residual correction model.
        
        Parameters:
          prcp: Precipitation series (mm/day).
          pet: Potential evapotranspiration series (mm/day).
          q_obs: Observed streamflow (mm/day).
          dates: Optional date array for DOY feature computation.
          warmup_days: Number of initialization days to exclude from calibration loss.
          
        Returns:
          Dictionary of training metrics for both physical GR4J and Hybrid-GR4J.
        """
        # Step 1: Calibrate underlying physical GR4J model if not already calibrated
        if not self.gr4j.calibrated:
            gr4j_metrics = self.gr4j.calibrate(prcp, pet, q_obs, warmup_days=warmup_days)
        else:
            sim_init = self.gr4j.simulate(prcp, pet)["q_sim"]
            mask_eval = (~np.isnan(q_obs)) & (q_obs >= 0.0)
            if warmup_days > 0 and len(mask_eval) > warmup_days:
                mask_eval[:warmup_days] = False
            gr4j_metrics = compute_hydrological_metrics(q_obs[mask_eval], sim_init[mask_eval])

        # Step 2: Simulate internal stores and discharge
        sim_res = self.gr4j.simulate(prcp, pet, return_stores=True)
        q_gr4j = sim_res["q_sim"]

        # Step 3: Extract multi-scale state features
        X_all, _ = self._extract_features(prcp, pet, dates=dates, gr4j_sim_res=sim_res)

        # Step 4: Compute physical residual target e_t = q_obs - q_gr4j
        residuals = q_obs - q_gr4j

        # Valid mask: valid observed streamflow, post-warmup, non-NaN
        valid_mask = (~np.isnan(q_obs)) & (q_obs >= 0.0) & (~np.isnan(residuals))
        if warmup_days > 0 and len(valid_mask) > warmup_days:
            valid_mask[:warmup_days] = False

        X_train = X_all[valid_mask]
        y_train = residuals[valid_mask]

        if len(X_train) < 50:
            # Insufficient points to fit residual model; fall back to pure GR4J
            self.calibrated = True
            return {"gr4j_metrics": gr4j_metrics, "hybrid_metrics": gr4j_metrics}

        # Step 5: Fit residual model
        X_train_scaled = self.scaler.fit_transform(X_train)

        if self.use_lgbm:
            try:
                import lightgbm as lgb
                self.residual_model = lgb.LGBMRegressor(
                    n_estimators=40,
                    max_depth=3,
                    learning_rate=0.05,
                    num_leaves=8,
                    min_child_samples=20,
                    random_state=42,
                    n_jobs=2,
                    verbose=-1,
                )
                self.residual_model.fit(X_train_scaled, y_train)
            except Exception:
                # Fallback to Ridge
                self.residual_model = Ridge(alpha=self.alpha)
                self.residual_model.fit(X_train_scaled, y_train)
        else:
            self.residual_model = Ridge(alpha=self.alpha)
            self.residual_model.fit(X_train_scaled, y_train)

        self.calibrated = True

        # Evaluate on training post-warmup period
        y_pred_res = self.residual_model.predict(X_train_scaled)
        q_hybrid_train = np.maximum(0.0, q_gr4j[valid_mask] + y_pred_res)
        hybrid_metrics = compute_hydrological_metrics(q_obs[valid_mask], q_hybrid_train)

        return {
            "gr4j_metrics": gr4j_metrics,
            "hybrid_metrics": hybrid_metrics,
            "kge_improvement": float(hybrid_metrics.get("kge", 0.0) - gr4j_metrics.get("kge", 0.0)),
        }

    def simulate(
        self,
        prcp: np.ndarray,
        pet: np.ndarray,
        dates: Optional[Any] = None,
        return_components: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Runs the hybrid simulation loop.
        
        Returns:
          Dictionary containing:
            - 'q_sim': Hybrid streamflow prediction (non-negative).
            - 'q_gr4j': Base physical GR4J simulation.
            - 'residual_correction': Additive residual correction term.
        """
        # Physical simulation
        sim_res = self.gr4j.simulate(prcp, pet, return_stores=True)
        q_gr4j = sim_res["q_sim"]

        if not self.calibrated or self.residual_model is None:
            # Pure GR4J fallback
            res = {"q_sim": q_gr4j, "q_gr4j": q_gr4j, "residual_correction": np.zeros_like(q_gr4j)}
            if return_components:
                res["s_store"] = sim_res["s_store"]
                res["r_store"] = sim_res["r_store"]
            return res

        # Feature extraction
        X_all, _ = self._extract_features(prcp, pet, dates=dates, gr4j_sim_res=sim_res)
        X_scaled = self.scaler.transform(X_all)

        # Residual inference
        residual_pred = self.residual_model.predict(X_scaled)

        # Non-negative fusion: max(0.0, Q_GR4J + delta)
        q_hybrid = np.maximum(0.0, q_gr4j + residual_pred)

        res = {
            "q_sim": q_hybrid,
            "q_gr4j": q_gr4j,
            "residual_correction": residual_pred,
        }
        if return_components:
            res["s_store"] = sim_res["s_store"]
            res["r_store"] = sim_res["r_store"]
        return res

    def to_dict(self) -> Dict[str, Any]:
        """Serializes HybridGR4J configuration and parameters."""
        data: Dict[str, Any] = {
            "gr4j_params": [float(p) for p in self.gr4j.params],
            "alpha": self.alpha,
            "use_lgbm": self.use_lgbm,
            "calibrated": self.calibrated,
            "feature_names": self.feature_names,
        }
        if self.calibrated and self.residual_model is not None:
            if hasattr(self.scaler, "mean_") and self.scaler.mean_ is not None:
                data["scaler_mean"] = [float(m) for m in self.scaler.mean_]
                data["scaler_scale"] = [float(s) for s in self.scaler.scale_]
            if isinstance(self.residual_model, Ridge):
                data["model_type"] = "ridge"
                data["intercept"] = float(self.residual_model.intercept_)
                data["coefficients"] = [float(c) for c in self.residual_model.coef_]
            else:
                data["model_type"] = "other"
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridGR4J":
        """Reconstructs HybridGR4J from a dictionary."""
        params = np.array(data["gr4j_params"], dtype=np.float64)
        model = cls(gr4j_params=params, alpha=data.get("alpha", 25.0), use_lgbm=data.get("use_lgbm", False))
        model.calibrated = data.get("calibrated", False)
        if model.calibrated and data.get("model_type") == "ridge":
            model.scaler.mean_ = np.array(data["scaler_mean"], dtype=np.float64)
            model.scaler.scale_ = np.array(data["scaler_scale"], dtype=np.float64)
            ridge = Ridge(alpha=model.alpha)
            ridge.intercept_ = float(data["intercept"])
            ridge.coef_ = np.array(data["coefficients"], dtype=np.float64)
            model.residual_model = ridge
        return model
