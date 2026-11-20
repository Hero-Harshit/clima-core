"""HeatRisk V5: Soft-Gated Regularized Mixture of Experts Engine.
Combines a universal global LightGBM backbone prior with 5 regional residual experts
and a softmax bioclimatic gating network to eliminate sample starvation and boundary discontinuities.
"""

import os
import json
import joblib
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, median_absolute_error


# Canonical 50-city bioclimatic zone dictionary
CITY_TO_ZONE: Dict[str, int] = {
    # Zone 0: Humid Tropical Coastal (10 cities)
    "mumbai": 0, "chennai": 0, "kochi": 0, "panaji": 0, "mangaluru": 0,
    "visakhapatnam": 0, "surat": 0, "thiruvananthapuram": 0, "puducherry": 0, "port blair": 0,
    # Zone 1: Deccan Plateau Corridor (7 cities)
    "bengaluru": 1, "hyderabad": 1, "pune": 1, "mysuru": 1, "coimbatore": 1,
    "madurai": 1, "vijayawada": 1,
    # Zone 2: Indo-Gangetic Plains & East (13 cities)
    "delhi": 2, "kolkata": 2, "patna": 2, "lucknow": 2, "kanpur": 2,
    "varanasi": 2, "agra": 2, "prayagraj": 2, "gwalior": 2, "bhopal": 2,
    "indore": 2, "ranchi": 2, "bhubaneswar": 2,
    # Zone 3: Semi-Arid & Central Corridor (10 cities)
    "jaipur": 3, "jodhpur": 3, "ahmedabad": 3, "vadodara": 3, "nagpur": 3,
    "raipur": 3, "dehradun": 3, "chandigarh": 3, "amritsar": 3, "ludhiana": 3,
    # Zone 4: Sub-Himalayan & Montane Northeast (10 cities)
    "shimla": 4, "srinagar": 4, "leh": 4, "jammu": 4, "gangtok": 4,
    "shillong": 4, "guwahati": 4, "agartala": 4, "imphal": 4, "nashik": 4,
}

ZONE_NAMES: Dict[int, str] = {
    0: "Humid Tropical Coastal",
    1: "Deccan Plateau Corridor",
    2: "Indo-Gangetic Plains & East",
    3: "Semi-Arid & Central Corridor",
    4: "Sub-Himalayan & Montane Northeast",
}

GATING_FEATURES: List[str] = [
    "latitude",
    "longitude",
    "elevation_m",
    "sin_day_of_year",
    "cos_day_of_year",
    "temperature_2m_mean_lag_1",
    "relative_humidity_2m_mean_lag_1",
]


class SoftGatedMoEV5:
    """Soft-Gated Regularized Mixture of Experts for Maximum Temperature Prediction.
    
    Formula:
      y_hat(x) = y_global(x) + lambda * sum_{k=0}^4 g_k(x) * Delta_y_k(x)
    where:
      y_global(x) is the global champion backbone prior.
      Delta_y_k(x) is the regional expert predicting the residual (y - y_global).
      g_k(x) is the softmax gating probability for zone k.
      lambda is the shrinkage parameter in [0, 1].
    """

    def __init__(self, lambda_shrinkage: float = 0.5):
        self.lambda_shrinkage = lambda_shrinkage
        self.global_backbone: Optional[lgb.Booster] = None
        self.experts_: Dict[int, lgb.Booster] = {}
        self.gating_network: Optional[LogisticRegression] = None
        self.gating_scaler: Optional[StandardScaler] = None
        self.feature_cols: List[str] = []

    def fit(
        self,
        df_train: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        global_booster_path: str,
    ) -> "SoftGatedMoEV5":
        self.feature_cols = feature_cols
        self.global_backbone = lgb.Booster(model_file=global_booster_path)
        
        # 1. Compute global backbone predictions on training data
        X_train_full = df_train[feature_cols].astype(np.float32).values
        y_train_full = df_train[target_col].astype(np.float32).values
        y_global_train = self.global_backbone.predict(X_train_full)
        residuals_train = y_train_full - y_global_train

        # Assign zones
        zones_train = df_train["city"].str.lower().map(CITY_TO_ZONE).fillna(2).astype(int).values

        # 2. Train Softmax Gating Network on bioclimatic features
        X_gate = df_train[GATING_FEATURES].astype(np.float64).values
        self.gating_scaler = StandardScaler()
        X_gate_scaled = self.gating_scaler.fit_transform(X_gate)
        
        self.gating_network = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            C=1.0,
            max_iter=500,
            random_state=42,
        )
        self.gating_network.fit(X_gate_scaled, zones_train)

        # 3. Train Regional Residual Experts
        expert_params = {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "n_estimators": 120,
            "learning_rate": 0.04,
            "num_leaves": 20,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": 2,
            "verbose": -1,
        }

        for k in range(5):
            mask_k = zones_train == k
            if np.sum(mask_k) > 50:
                X_k = X_train_full[mask_k]
                res_k = residuals_train[mask_k]
                
                exp_model = lgb.LGBMRegressor(**expert_params)
                exp_model.fit(X_k, res_k)
                self.experts_[k] = exp_model.booster_
            else:
                self.experts_[k] = None

        return self

    def predict_components(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        X = df[self.feature_cols].astype(np.float32).values
        
        # Global prior
        y_global = self.global_backbone.predict(X)
        
        # Gating probabilities
        X_gate = df[GATING_FEATURES].astype(np.float64).values
        X_gate_scaled = self.gating_scaler.transform(X_gate)
        gating_probs = self.gating_network.predict_proba(X_gate_scaled)  # Shape (N, 5)
        
        # Residual expert corrections
        N = len(df)
        residual_corrections = np.zeros(N, dtype=np.float64)
        for k in range(5):
            if self.experts_.get(k) is not None:
                pred_res_k = self.experts_[k].predict(X)
                residual_corrections += gating_probs[:, k] * pred_res_k

        y_moe = y_global + self.lambda_shrinkage * residual_corrections

        return {
            "y_global": y_global,
            "y_moe": y_moe,
            "gating_probs": gating_probs,
            "residual_corrections": residual_corrections,
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_components(df)["y_moe"]
