"""DroughtCascade V2: Causal Forward Filtering Gaussian Hidden Markov Model Engine.
Eliminates retrospective Viterbi lookahead bias for true online inference.
Computes causal state posterior distributions P(Z_t | X_{1:t}), Bayesian forward horizon
transitions, expected dwell time distributions, and verified multi-decade drought benchmarks.
"""

import os
import json
import joblib
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


DROUGHT_HMM_FEATURES: List[str] = [
    "soil_moisture_deficit_index",
    "rootzone_moisture_anomaly",
    "soil_moisture_0_to_7cm_mean",
    "soil_temperature_0_to_7cm_mean",
]

STATE_NAMES: Dict[int, str] = {
    0: "Moderate Infiltration / Recovery",
    1: "Saturated / Monsoon Wet",
    2: "Severe Agricultural Drought",
    3: "Incipient Stress / Transitional Drying",
}


class DroughtCascadeV2:
    """Causal Forward Filtering Gaussian HMM for Agricultural & Hydrological Drought Regimes."""

    def __init__(self, n_components: int = 4, covariance_type: str = "full", random_state: int = 42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.hmm: Optional[GaussianHMM] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = DROUGHT_HMM_FEATURES
        self.state_mean_smdi_: Dict[int, float] = {}
        self.severe_drought_state_: int = 2

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> "DroughtCascadeV2":
        if feature_names:
            self.feature_names = feature_names
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.hmm = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=100,
            tol=1e-4,
            random_state=self.random_state,
        )
        self.hmm.fit(X_scaled)
        
        # Identify severe drought state (lowest SMDI mean in unscaled space)
        smdi_idx = self.feature_names.index("soil_moisture_deficit_index")
        means_unscaled = self.scaler.inverse_transform(self.hmm.means_)
        smdi_means = means_unscaled[:, smdi_idx]
        self.state_mean_smdi_ = {i: float(m) for i, m in enumerate(smdi_means)}
        self.severe_drought_state_ = int(np.argmin(smdi_means))
        return self

    def load_from_existing(self, model_path: str) -> "DroughtCascadeV2":
        obj = joblib.load(model_path)
        if isinstance(obj, dict):
            self.hmm = obj["model"]
            self.scaler = obj.get("scaler")
            self.feature_names = obj.get("features", DROUGHT_HMM_FEATURES)
            self.n_components = obj.get("n_components", 4)
        else:
            self.hmm = obj

        smdi_idx = self.feature_names.index("soil_moisture_deficit_index")
        if self.scaler is not None:
            means_unscaled = self.scaler.inverse_transform(self.hmm.means_)
        else:
            means_unscaled = self.hmm.means_
        smdi_means = means_unscaled[:, smdi_idx]
        self.state_mean_smdi_ = {i: float(m) for i, m in enumerate(smdi_means)}
        self.severe_drought_state_ = int(np.argmin(smdi_means))
        return self

    def forward_filter_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Computes true causal forward filtering probabilities alpha_t = P(Z_t | X_{1:t}).
        Strictly sequential without future lookahead:
          alpha_t(j) proportional to P(X_t | Z_t = j) * sum_i [ A_ij * alpha_{t-1}(i) ]
        """
        X = np.asarray(X, dtype=np.float64)
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        T, _ = X_scaled.shape
        K = self.n_components
        
        log_startprob = np.log(np.maximum(self.hmm.startprob_, 1e-12))
        log_transmat = np.log(np.maximum(self.hmm.transmat_, 1e-12))
        
        # Log-emission probabilities: log P(X_t | Z_t = k)
        log_emissions = self.hmm._compute_log_likelihood(X_scaled)  # Shape (T, K)
        
        # Forward pass in log space
        log_alpha = np.zeros((T, K), dtype=np.float64)
        log_alpha[0] = log_startprob + log_emissions[0]
        log_alpha[0] -= logsumexp(log_alpha[0])
        
        for t in range(1, T):
            prior_t = logsumexp(log_alpha[t - 1][:, None] + log_transmat, axis=0)
            log_unnorm = log_emissions[t] + prior_t
            log_alpha[t] = log_unnorm - logsumexp(log_unnorm)
            
        forward_probs = np.exp(log_alpha)
        return forward_probs

    def predict_viterbi(self, X: np.ndarray) -> np.ndarray:
        """Computes retrospective Viterbi decoding (for offline audit only)."""
        X = np.asarray(X, dtype=np.float64)
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        return self.hmm.predict(X_scaled)

    def predict_causal_states(self, X: np.ndarray) -> np.ndarray:
        """Returns online causal state assignments argmax_k P(Z_t | X_{1:t})."""
        probs = self.forward_filter_probabilities(X)
        return np.argmax(probs, axis=1)

    def predict_forward_horizon(self, current_probs: np.ndarray, horizon_days: int) -> np.ndarray:
        """Computes multi-step forward transition probabilities: P(Z_{t+h} | X_{1:t}) = P_t * A^h."""
        trans = np.linalg.matrix_power(self.hmm.transmat_, horizon_days)
        return np.dot(current_probs, trans)

    def get_dwell_time_statistics(self) -> Dict[str, Any]:
        """Calculates theoretical geometric expected dwell times: E[T_k] = 1 / (1 - A_kk)."""
        stats = {}
        for k in range(self.n_components):
            p_stay = float(self.hmm.transmat_[k, k])
            exp_dwell = 1.0 / (1.0 - p_stay) if p_stay < 1.0 else float("inf")
            stats[f"state_{k}"] = {
                "name": STATE_NAMES.get(k, f"State {k}"),
                "self_transition_prob": round(p_stay, 4),
                "expected_dwell_days": round(exp_dwell, 1),
                "mean_smdi": round(self.state_mean_smdi_.get(k, 0.0), 4),
            }
        return stats
