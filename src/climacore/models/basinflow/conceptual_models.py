"""Conceptual Hydrological Models: GR4J and HBV-Light.
Implements the exact published equations for:
1. GR4J (Perrin et al., 2003): 4-parameter lumped hydrological model
2. HBV-Light (Bergström 1992; Seibert & Vis 2012): 9-parameter conceptual catchment model
With built-in KGE objective function calibration.
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.climacore.models.basinflow.metrics import compute_hydrological_metrics


# -----------------------------------------------------------------------------
# GR4J Implementation (Perrin et al., 2003)
# -----------------------------------------------------------------------------

class GR4J:
    """4-parameter lumped conceptual hydrological model (Perrin et al., 2003).
    
    Parameters:
      x1: Maximum capacity of the production store (soil moisture) [mm] (>0)
      x2: Groundwater exchange coefficient [mm/day] (unconstrained, positive or negative)
      x3: One-day ahead maximum capacity of the routing store [mm] (>0)
      x4: Time base of unit hydrographs UH1 and UH2 [days] (>0.5)
    """

    def __init__(
        self,
        x1: float = 350.0,
        x2: float = 0.0,
        x3: float = 90.0,
        x4: float = 1.5,
    ):
        self.params = np.array([x1, x2, x3, x4], dtype=np.float64)
        self.param_names = ["x1", "x2", "x3", "x4"]
        self.calibrated = False

    @staticmethod
    def _build_uh(x4: float) -> Tuple[np.ndarray, np.ndarray]:
        """Builds ordinates for UH1 (time base x4) and UH2 (time base 2*x4)."""
        x4 = max(float(x4), 0.5)
        n1 = int(np.ceil(x4))
        n2 = int(np.ceil(2.0 * x4))

        # UH1 S-curve
        t1 = np.arange(n1 + 1, dtype=np.float64)
        sh1 = np.zeros(n1 + 1, dtype=np.float64)
        for i, t in enumerate(t1):
            if t <= 0.0:
                sh1[i] = 0.0
            elif t < x4:
                sh1[i] = (t / x4) ** 2.5
            else:
                sh1[i] = 1.0
        uh1 = np.diff(sh1)
        uh1_sum = np.sum(uh1)
        if uh1_sum > 0:
            uh1 = uh1 / uh1_sum

        # UH2 S-curve
        t2 = np.arange(n2 + 1, dtype=np.float64)
        sh2 = np.zeros(n2 + 1, dtype=np.float64)
        for i, t in enumerate(t2):
            if t <= 0.0:
                sh2[i] = 0.0
            elif t < x4:
                sh2[i] = 0.5 * ((t / x4) ** 2.5)
            elif t < 2.0 * x4:
                sh2[i] = 1.0 - 0.5 * ((2.0 - t / x4) ** 2.5)
            else:
                sh2[i] = 1.0
        uh2 = np.diff(sh2)
        uh2_sum = np.sum(uh2)
        if uh2_sum > 0:
            uh2 = uh2 / uh2_sum

        return uh1, uh2

    def simulate(
        self,
        prcp: np.ndarray,
        pet: np.ndarray,
        params: Optional[np.ndarray] = None,
        return_stores: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Runs daily simulation loop of GR4J."""
        if params is None:
            params = self.params
        x1, x2, x3, x4 = params
        x1 = max(float(x1), 1.0)
        x3 = max(float(x3), 1.0)
        x4 = max(float(x4), 0.5)

        uh1, uh2 = self._build_uh(x4)
        n1 = len(uh1)
        n2 = len(uh2)

        n_steps = len(prcp)
        q_sim = np.zeros(n_steps, dtype=np.float64)
        s_store = np.zeros(n_steps, dtype=np.float64)
        r_store = np.zeros(n_steps, dtype=np.float64)

        # Initial states
        s = 0.3 * x1
        r = 0.5 * x3

        # UH convolution state queues
        uh1_state = np.zeros(n1, dtype=np.float64)
        uh2_state = np.zeros(n2, dtype=np.float64)

        for t in range(n_steps):
            p = float(prcp[t])
            e = float(pet[t])

            # 1. Net rainfall and net evapotranspiration
            if p > e:
                pn = p - e
                en = 0.0
                # Reservoir filling
                scaled_pn = min(pn / x1, 13.0)
                tanh_pn = np.tanh(scaled_pn)
                s_ratio = s / x1
                ps = (x1 * (1.0 - s_ratio ** 2) * tanh_pn) / (1.0 + s_ratio * tanh_pn)
                s += ps
                es = 0.0
            else:
                pn = 0.0
                en = e - p
                # Reservoir emptying
                scaled_en = min(en / x1, 13.0)
                tanh_en = np.tanh(scaled_en)
                s_ratio = s / x1
                es = (s * (2.0 - s_ratio) * tanh_en) / (1.0 + (1.0 - s_ratio) * tanh_en)
                s = max(0.0, s - es)
                ps = 0.0

            # 2. Percolation from production store
            s_ratio = s / x1
            perc_term = (4.0 / 9.0 * s_ratio) ** 4
            perc = s * (1.0 - (1.0 + perc_term) ** (-0.25))
            s = max(0.0, s - perc)

            # 3. Effective rainfall reaching routing
            pr = perc + (pn - ps)

            # 4. Convolution with UH1 (90%) and UH2 (10%)
            uh1_state += 0.9 * pr * uh1
            q9 = uh1_state[0]
            uh1_state = np.roll(uh1_state, -1)
            uh1_state[-1] = 0.0

            uh2_state += 0.1 * pr * uh2
            q1 = uh2_state[0]
            uh2_state = np.roll(uh2_state, -1)
            uh2_state[-1] = 0.0

            # 5. Groundwater exchange term
            r_ratio = max(0.0, r / x3)
            f_exchange = x2 * (r_ratio ** 3.5)

            # 6. Routing store update
            r = max(0.0, r + q9 + f_exchange)
            r_ratio = r / x3
            qr = r * (1.0 - (1.0 + (r_ratio ** 4)) ** (-0.25))
            r = max(0.0, r - qr)

            # 7. Direct flow
            qd = max(0.0, q1 + f_exchange)

            # 8. Total streamflow
            q_sim[t] = qr + qd
            s_store[t] = s
            r_store[t] = r

        res = {"q_sim": q_sim}
        if return_stores:
            res["s_store"] = s_store
            res["r_store"] = r_store
        return res

    def calibrate(
        self,
        prcp: np.ndarray,
        pet: np.ndarray,
        q_obs: np.ndarray,
        warmup_days: int = 365,
    ) -> Dict[str, Any]:
        """Calibrates GR4J parameters maximizing KGE on observed streamflow."""
        valid_obs = (~np.isnan(q_obs)) & (q_obs >= 0.0)

        # Transformation to unconstrained space for Nelder-Mead / L-BFGS
        # bounds: x1 in [10, 2000], x2 in [-5, 5], x3 in [10, 1000], x4 in [0.5, 10.0]
        bounds = [(10.0, 2500.0), (-6.0, 6.0), (10.0, 1200.0), (0.5, 10.0)]
        init_guess = np.array([350.0, 0.0, 90.0, 1.8], dtype=np.float64)

        def objective(p):
            # Check bounds penalty
            if not (bounds[0][0] <= p[0] <= bounds[0][1] and
                    bounds[1][0] <= p[1] <= bounds[1][1] and
                    bounds[2][0] <= p[2] <= bounds[2][1] and
                    bounds[3][0] <= p[3] <= bounds[3][1]):
                return 999.0

            sim = self.simulate(prcp, pet, params=p)["q_sim"]
            # Evaluate after warm-up
            mask = valid_obs.copy()
            if warmup_days > 0:
                mask[:warmup_days] = False
            
            if np.sum(mask) < 30:
                return 999.0

            o = q_obs[mask].astype(np.float64)
            s = sim[mask].astype(np.float64)

            # KGE loss
            o_std = np.std(o)
            s_std = np.std(s)
            o_mean = np.mean(o)
            s_mean = np.mean(s)

            if o_std < 1e-6 or s_std < 1e-6 or o_mean < 1e-6:
                return 999.0

            r = float(np.corrcoef(o, s)[0, 1])
            if np.isnan(r):
                return 999.0
            alpha = s_std / o_std
            beta = s_mean / o_mean
            kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
            return -float(kge)

        res = minimize(
            objective,
            init_guess,
            method="Nelder-Mead",
            options={"maxiter": 300, "xatol": 0.05, "fatol": 0.005},
        )

        self.params = res.x
        self.calibrated = True

        final_sim = self.simulate(prcp, pet)["q_sim"]
        metrics = compute_hydrological_metrics(q_obs[warmup_days:], final_sim[warmup_days:])
        
        return {
            "success": bool(res.success),
            "calibrated_params": {name: round(float(val), 3) for name, val in zip(self.param_names, self.params)},
            "train_metrics": metrics,
        }

    def predict(self, prcp: np.ndarray, pet: np.ndarray) -> np.ndarray:
        return self.simulate(prcp, pet)["q_sim"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": "GR4J",
            "params": {name: float(val) for name, val in zip(self.param_names, self.params)},
            "calibrated": self.calibrated,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GR4J":
        p = d["params"]
        m = cls(x1=p["x1"], x2=p["x2"], x3=p["x3"], x4=p["x4"])
        m.calibrated = d.get("calibrated", False)
        return m


# -----------------------------------------------------------------------------
# HBV-Light Implementation (Bergström 1992; Seibert & Vis 2012)
# -----------------------------------------------------------------------------

class HBVLight:
    """Conceptual Hydrological Model HBV-Light (Bergström 1992, Seibert & Vis 2012).
    
    Parameters:
      fc: Maximum soil moisture storage [mm] (>0)
      beta: Shape parameter for runoff generation (>0)
      lp: Limit for potential evapotranspiration fraction (0 < lp <= 1)
      perc: Maximum percolation rate to lower zone [mm/day] (>=0)
      uzl: Threshold storage for upper fast outflow [mm] (>=0)
      k0: Recession coefficient for additional upper outflow [1/day] (>=0)
      k1: Recession coefficient for upper outflow [1/day] (>=0)
      k2: Recession coefficient for lower outflow (baseflow) [1/day] (>=0)
      maxbas: Base length of triangular weighting function [days] (>=1)
    """

    def __init__(
        self,
        fc: float = 300.0,
        beta: float = 2.0,
        lp: float = 0.7,
        perc: float = 1.5,
        uzl: float = 25.0,
        k0: float = 0.2,
        k1: float = 0.1,
        k2: float = 0.02,
        maxbas: float = 2.0,
    ):
        self.params = np.array([fc, beta, lp, perc, uzl, k0, k1, k2, maxbas], dtype=np.float64)
        self.param_names = ["fc", "beta", "lp", "perc", "uzl", "k0", "k1", "k2", "maxbas"]
        self.calibrated = False

    @staticmethod
    def _triangular_weights(maxbas: float) -> np.ndarray:
        """Calculates triangular routing weighting filter ordinates."""
        maxbas = max(float(maxbas), 1.0)
        n = int(np.ceil(maxbas))
        weights = np.zeros(n, dtype=np.float64)
        h = 2.0 / maxbas
        half = maxbas / 2.0

        for i in range(1, n + 1):
            t_left = i - 1.0
            t_right = min(float(i), maxbas)
            # Integrate triangle over [t_left, t_right]
            def f(u):
                if u <= half:
                    return h * (u / half)
                elif u <= maxbas:
                    return h * (1.0 - (u - half) / half)
                return 0.0

            # Midpoint approximation of integral
            mid = 0.5 * (t_left + t_right)
            weights[i - 1] = f(mid) * (t_right - t_left)

        w_sum = np.sum(weights)
        if w_sum > 0:
            weights = weights / w_sum
        return weights

    def simulate(
        self,
        prcp: np.ndarray,
        pet: np.ndarray,
        params: Optional[np.ndarray] = None,
        return_stores: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Runs daily simulation loop of HBV-Light."""
        if params is None:
            params = self.params
        fc, beta, lp, perc, uzl, k0, k1, k2, maxbas = params
        fc = max(float(fc), 10.0)
        beta = max(float(beta), 0.5)
        lp = np.clip(float(lp), 0.1, 1.0)
        perc = max(float(perc), 0.0)
        uzl = max(float(uzl), 0.0)
        k0 = max(float(k0), 0.0)
        k1 = max(float(k1), 0.0)
        k2 = max(float(k2), 0.0)
        maxbas = max(float(maxbas), 1.0)

        weights = self._triangular_weights(maxbas)
        n_weights = len(weights)

        n_steps = len(prcp)
        q_gen = np.zeros(n_steps, dtype=np.float64)
        q_sim = np.zeros(n_steps, dtype=np.float64)
        s_soil = np.zeros(n_steps, dtype=np.float64)
        s_suz = np.zeros(n_steps, dtype=np.float64)
        s_slz = np.zeros(n_steps, dtype=np.float64)

        # Initial storages
        sm = 0.5 * fc
        suz = 5.0
        slz = 10.0

        for t in range(n_steps):
            p = float(prcp[t])
            e = float(pet[t])

            # 1. Soil moisture routine
            # Runoff generation (recharge to upper groundwater store)
            ratio = min(sm / fc, 1.0)
            recharge = p * (ratio ** beta)
            sm = min(fc, sm + p - recharge)

            # Evapotranspiration
            if sm >= lp * fc:
                ea = e
            else:
                ea = e * (sm / (lp * fc))
            ea = min(ea, sm)
            sm = max(0.0, sm - ea)

            # 2. Upper groundwater box (SUZ)
            suz += recharge
            actual_perc = min(perc, suz)
            suz -= actual_perc

            # 3. Lower groundwater box (SLZ)
            slz += actual_perc

            # 4. Outflows from groundwater boxes
            # Upper box
            q0 = k0 * max(0.0, suz - uzl)
            q1 = k1 * suz
            q_upper = min(q0 + q1, suz)
            suz = max(0.0, suz - q_upper)

            # Lower box (baseflow)
            q2 = min(k2 * slz, slz)
            slz = max(0.0, slz - q2)

            q_gen[t] = q_upper + q2
            s_soil[t] = sm
            s_suz[t] = suz
            s_slz[t] = slz

        # 5. Routing using triangular filter convolution
        q_sim = np.convolve(q_gen, weights, mode="full")[:n_steps]

        res = {"q_sim": q_sim}
        if return_stores:
            res["soil_store"] = s_soil
            res["upper_store"] = s_suz
            res["lower_store"] = s_slz
        return res

    def calibrate(
        self,
        prcp: np.ndarray,
        pet: np.ndarray,
        q_obs: np.ndarray,
        warmup_days: int = 365,
    ) -> Dict[str, Any]:
        """Calibrates HBV-Light parameters maximizing KGE."""
        valid_obs = (~np.isnan(q_obs)) & (q_obs >= 0.0)

        bounds = [
            (50.0, 800.0),    # fc
            (1.0, 5.0),       # beta
            (0.3, 1.0),       # lp
            (0.1, 5.0),       # perc
            (5.0, 80.0),      # uzl
            (0.05, 0.5),      # k0
            (0.01, 0.3),      # k1
            (0.001, 0.1),     # k2
            (1.0, 4.5),       # maxbas
        ]
        init_guess = np.array([250.0, 2.0, 0.7, 1.5, 25.0, 0.15, 0.08, 0.02, 2.0], dtype=np.float64)

        def objective(p):
            for i, (b_low, b_high) in enumerate(bounds):
                if not (b_low <= p[i] <= b_high):
                    return 999.0

            sim = self.simulate(prcp, pet, params=p)["q_sim"]
            mask = valid_obs.copy()
            if warmup_days > 0:
                mask[:warmup_days] = False
            
            if np.sum(mask) < 30:
                return 999.0

            o = q_obs[mask].astype(np.float64)
            s = sim[mask].astype(np.float64)

            o_std = np.std(o)
            s_std = np.std(s)
            o_mean = np.mean(o)
            s_mean = np.mean(s)

            if o_std < 1e-6 or s_std < 1e-6 or o_mean < 1e-6:
                return 999.0

            r = float(np.corrcoef(o, s)[0, 1])
            if np.isnan(r):
                return 999.0
            alpha = s_std / o_std
            beta = s_mean / o_mean
            kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
            return -float(kge)

        res = minimize(
            objective,
            init_guess,
            method="Nelder-Mead",
            options={"maxiter": 350, "xatol": 0.05, "fatol": 0.005},
        )

        self.params = res.x
        self.calibrated = True

        final_sim = self.simulate(prcp, pet)["q_sim"]
        metrics = compute_hydrological_metrics(q_obs[warmup_days:], final_sim[warmup_days:])

        return {
            "success": bool(res.success),
            "calibrated_params": {name: round(float(val), 4) for name, val in zip(self.param_names, self.params)},
            "train_metrics": metrics,
        }

    def predict(self, prcp: np.ndarray, pet: np.ndarray) -> np.ndarray:
        return self.simulate(prcp, pet)["q_sim"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": "HBV-Light",
            "params": {name: float(val) for name, val in zip(self.param_names, self.params)},
            "calibrated": self.calibrated,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HBVLight":
        p = d["params"]
        m = cls(
            fc=p["fc"], beta=p["beta"], lp=p["lp"], perc=p["perc"],
            uzl=p["uzl"], k0=p["k0"], k1=p["k1"], k2=p["k2"], maxbas=p["maxbas"]
        )
        m.calibrated = d.get("calibrated", False)
        return m
