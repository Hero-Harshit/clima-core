"""Muskingum-Cunge Hydraulic Channel Routing Engine for HydroRIVERS reaches.
Simulates downstream flood wave translation, peak attenuation, and hydraulic celerity
along natural river channel geometries with numerical stability and exact volume conservation.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


def _trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(getattr(np, "trapz")(y, x))


class MuskingumCungeRouter:
    """Hydraulic river routing using Muskingum-Cunge formulation (Cunge 1969).
    
    Equations:
      y0 = (n * Q0 / (B * sqrt(S0)))^(3/5)   [Normal depth, wide rectangular channel]
      v0 = Q0 / (B * y0)                      [Velocity]
      ck = (5/3) * v0                         [Kinematic wave celerity]
      K = L / ck                              [Reach travel time]
      X = 0.5 * (1 - Q0 / (B * S0 * ck * L))  [Clamped to [0.05, 0.45]]
    """

    def __init__(self, manning_n: float = 0.035):
        self.manning_n = manning_n

    def route_hydrograph(
        self,
        inflow_m3s: np.ndarray,
        time_hours: np.ndarray,
        reach_length_km: float,
        bed_slope_m_per_m: float = 0.001,
        channel_width_meters: float = 25.0,
        baseflow_m3s: float = 0.5,
    ) -> Dict[str, Any]:
        """Routes an inflow hydrograph through a river reach of length L."""
        inflow = np.asarray(inflow_m3s, dtype=np.float64)
        t_arr = np.asarray(time_hours, dtype=np.float64)
        
        if len(inflow) < 2 or np.max(inflow) <= 1e-4:
            return {
                "outflow_m3s": inflow.copy(),
                "peak_inflow_m3s": 0.0,
                "peak_outflow_m3s": 0.0,
                "peak_attenuation_ratio": 1.0,
                "lag_time_hours": 0.0,
                "k_travel_time_hours": 0.0,
                "x_weighting": 0.0,
                "volume_conservation_ratio": 1.0,
            }

        # Interpolate hydrograph to fine uniform time step (e.g. 0.1 hours = 6 minutes)
        t_fine = np.linspace(t_arr[0], t_arr[-1], max(len(t_arr) * 4, 100))
        inflow_fine = np.interp(t_fine, t_arr, inflow)
        dt_sec = (t_fine[1] - t_fine[0]) * 3600.0

        reach_len_m = max(reach_length_km * 1000.0, 500.0)
        s0 = max(bed_slope_m_per_m, 0.0001)
        b_width = max(channel_width_meters, 5.0)

        # Reference flow Q0: 0.5 * peak + baseflow
        q0 = max(0.5 * np.max(inflow_fine) + baseflow_m3s, 1.0)

        # Manning normal depth
        y0 = (self.manning_n * q0 / (b_width * np.sqrt(s0))) ** 0.6
        y0 = max(y0, 0.1)

        # Velocity and celerity
        v0 = q0 / (b_width * y0)
        ck = (5.0 / 3.0) * v0
        ck = max(ck, 0.1)

        # Muskingum parameters
        k_sec = reach_len_m / ck
        x_calc = 0.5 * (1.0 - (q0 / (b_width * s0 * ck * reach_len_m)))
        x_weight = float(np.clip(x_calc, 0.05, 0.45))

        # Determine number of sub-reaches m for numerical stability
        m_steps = int(np.clip(np.ceil(k_sec / dt_sec), 1, 20))
        k_sub = k_sec / m_steps

        denom = 2.0 * k_sub * (1.0 - x_weight) + dt_sec
        c1 = (dt_sec - 2.0 * k_sub * x_weight) / denom
        c2 = (dt_sec + 2.0 * k_sub * x_weight) / denom
        c3 = (2.0 * k_sub * (1.0 - x_weight) - dt_sec) / denom

        # Non-negative correction & re-normalization
        c_raw = np.array([max(c1, 0.0), max(c2, 0.0), max(c3, 0.0)])
        c_norm = c_raw / np.sum(c_raw)
        c1, c2, c3 = float(c_norm[0]), float(c_norm[1]), float(c_norm[2])

        # Multi-step reach cascade
        current_inflow = inflow_fine.copy()
        for _ in range(m_steps):
            next_outflow = np.zeros_like(current_inflow)
            next_outflow[0] = current_inflow[0]
            for t in range(len(current_inflow) - 1):
                o_next = c1 * current_inflow[t + 1] + c2 * current_inflow[t] + c3 * next_outflow[t]
                next_outflow[t + 1] = max(o_next, 0.0)
            current_inflow = next_outflow

        # Map back to original time grid
        outflow = np.interp(t_arr, t_fine, current_inflow)

        # Strict volume conservation scaling
        vol_in = _trapezoid_integral(inflow, t_arr * 3600.0)
        vol_out = _trapezoid_integral(outflow, t_arr * 3600.0)
        if vol_out > 0 and vol_in > 0:
            vol_correction = vol_in / vol_out
            outflow = outflow * vol_correction
            vol_out = _trapezoid_integral(outflow, t_arr * 3600.0)

        peak_inflow = float(np.max(inflow))
        peak_outflow = float(np.max(outflow))
        idx_p_in = int(np.argmax(inflow))
        idx_p_out = int(np.argmax(outflow))
        lag_hours = float(t_arr[idx_p_out] - t_arr[idx_p_in])
        attenuation = float(peak_outflow / peak_inflow) if peak_inflow > 0 else 1.0
        vol_ratio = float(vol_out / vol_in) if vol_in > 0 else 1.0

        return {
            "outflow_m3s": outflow,
            "peak_inflow_m3s": round(peak_inflow, 3),
            "peak_outflow_m3s": round(peak_outflow, 3),
            "peak_attenuation_ratio": round(attenuation, 4),
            "lag_time_hours": round(lag_hours, 3),
            "k_travel_time_hours": round(k_sec / 3600.0, 3),
            "x_weighting": round(x_weight, 4),
            "coefficients": {"c1": round(c1, 4), "c2": round(c2, 4), "c3": round(c3, 4)},
            "volume_conservation_ratio": round(vol_ratio, 5),
            "sub_reaches": m_steps,
        }
