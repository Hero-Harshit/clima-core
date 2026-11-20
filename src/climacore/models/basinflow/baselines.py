"""Baseline and Physical Benchmark Models for BasinFlow V2.
Implements:
1. Persistence Baseline (Q_t = Q_{t-1})
2. Seasonal Climatology Baseline (Q_t = mean(Q_DOY))
3. SCS-CN Runoff Baseline
4. BasinFlow V1 SCS Unit Hydrograph routing baseline
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.climacore.models.basinflow.metrics import compute_hydrological_metrics
from src.climacore.models.basinflow.unit_hydrograph import SCSUnitHydrograph


class PersistenceBaseline:
    """Persistence model predicting streamflow as the most recent known observation."""

    def __init__(self):
        self.last_val = 0.0

    def fit(self, q_train: np.ndarray):
        valid = q_train[(~np.isnan(q_train)) & (q_train >= 0.0)]
        self.last_val = float(np.median(valid)) if len(valid) > 0 else 0.0

    def predict(self, q_obs: np.ndarray) -> np.ndarray:
        """Generates 1-day ahead persistence predictions."""
        n = len(q_obs)
        pred = np.zeros(n, dtype=np.float64)
        curr = self.last_val
        for t in range(n):
            pred[t] = curr
            val = q_obs[t]
            if not np.isnan(val) and val >= 0.0:
                curr = float(val)
        return pred


class SeasonalClimatologyBaseline:
    """Seasonal Climatology predicting mean historical streamflow for each day of year."""

    def __init__(self):
        self.doy_means = np.zeros(367, dtype=np.float64)
        self.global_mean = 0.0

    def fit(self, dates, q_train: np.ndarray):
        dt_idx = pd.to_datetime(dates)
        doy = dt_idx.dt.dayofyear.values if hasattr(dt_idx, "dt") else dt_idx.dayofyear.values
        df = pd.DataFrame({"doy": doy, "q": q_train})
        valid = df[(~df["q"].isna()) & (df["q"] >= 0.0)]
        if len(valid) == 0:
            return
        self.global_mean = float(valid["q"].mean())
        grouped = valid.groupby("doy")["q"].mean()
        for d, val in grouped.items():
            if 1 <= d <= 366:
                self.doy_means[d] = float(val)

        # Impute missing days of year with global mean
        zero_mask = self.doy_means == 0.0
        self.doy_means[zero_mask] = self.global_mean

    def predict(self, dates) -> np.ndarray:
        dt_idx = pd.to_datetime(dates)
        doys = dt_idx.dt.dayofyear.values if hasattr(dt_idx, "dt") else dt_idx.dayofyear.values
        preds = np.array([self.doy_means[min(int(d), 366)] for d in doys], dtype=np.float64)
        return preds


class SCSCNBaseline:
    """Deterministic SCS Curve Number event runoff model (USDA NRCS NEH-4)."""

    def __init__(self, cn: float = 75.0, ia_ratio: float = 0.2):
        self.cn = cn
        self.ia_ratio = ia_ratio

    def _calc_runoff(self, prcp: np.ndarray, cn: float) -> np.ndarray:
        s_val = (25400.0 / cn) - 254.0
        ia = self.ia_ratio * s_val
        q = np.where(prcp > ia, ((prcp - ia) ** 2) / (prcp - ia + s_val), 0.0)
        return q.astype(np.float64)

    def calibrate(self, prcp: np.ndarray, q_obs: np.ndarray) -> Dict[str, Any]:
        """Calibrates CN to minimize KGE error against observed daily runoff."""
        valid_mask = (~np.isnan(q_obs)) & (q_obs >= 0.0)

        def loss(cn):
            sim = self._calc_runoff(prcp, cn)
            o = q_obs[valid_mask]
            s = sim[valid_mask]
            if np.std(s) < 1e-6:
                return 999.0
            r = float(np.corrcoef(o, s)[0, 1])
            alpha = np.std(s) / np.std(o)
            beta = np.mean(s) / np.mean(o) if np.mean(o) > 0 else 1.0
            kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
            return -float(kge)

        res = minimize_scalar(loss, bounds=(40.0, 98.0), method="bounded")
        self.cn = float(res.x)
        sim = self._calc_runoff(prcp, self.cn)
        metrics = compute_hydrological_metrics(q_obs, sim)
        return {"cn": round(self.cn, 2), "metrics": metrics}

    def predict(self, prcp: np.ndarray) -> np.ndarray:
        return self._calc_runoff(prcp, self.cn)


class BasinFlowV1Baseline:
    """BasinFlow V1: SCS-CN Runoff Generation combined with Unit Hydrograph Routing."""

    def __init__(self, cn: float = 75.0, tc_hours: float = 12.0):
        self.cn = cn
        self.tc_hours = tc_hours
        self.scs = SCSCNBaseline(cn=cn)
        self.uh = SCSUnitHydrograph(delta_t_hours=24.0)

    def calibrate(self, prcp: np.ndarray, q_obs: np.ndarray) -> Dict[str, Any]:
        cal = self.scs.calibrate(prcp, q_obs)
        self.cn = cal["cn"]
        return {"cn": self.cn, "tc_hours": self.tc_hours}

    def predict(self, prcp: np.ndarray, area_km2: float = 1000.0) -> np.ndarray:
        # 1. Excess runoff depth
        q_excess = self.scs.predict(prcp)
        # 2. Daily linear reservoir / unit routing filter (t_lag ~ 1 to 2 days)
        # Triangular 2-day hydrograph filter for daily discretization
        routing_filter = np.array([0.2, 0.5, 0.3], dtype=np.float64)
        q_routed = np.convolve(q_excess, routing_filter, mode="full")[:len(prcp)]
        return q_routed
