"""Comprehensive Hydrological Evaluation Metrics for BasinFlow V2.
Implements standard metrics: NSE, KGE, LogNSE, RMSE, MAE, FHV, FLV,
peak timing error, peak magnitude error, and water balance error.
"""

from typing import Dict, Any
import numpy as np


def compute_hydrological_metrics(
    q_obs: np.ndarray,
    q_sim: np.ndarray,
    epsilon: float = 0.01,
) -> Dict[str, float]:
    """Calculates full suite of 10 benchmark hydrological metrics.
    
    Parameters
    ----------
    q_obs : 1D array of observed streamflow (mm/day or m3/s)
    q_sim : 1D array of simulated streamflow (matching units)
    epsilon : small positive offset for log-based low flow metrics
    
    Returns
    -------
    dict of metric name -> float value
    """
    # Filter valid observations (obs not NaN, obs >= 0)
    valid = (~np.isnan(q_obs)) & (~np.isnan(q_sim)) & (q_obs >= 0.0)
    if np.sum(valid) < 10:
        return {
            "nse": -999.0,
            "kge": -999.0,
            "log_nse": -999.0,
            "rmse": 999.0,
            "mae": 999.0,
            "fhv_pct": 999.0,
            "flv_pct": 999.0,
            "peak_timing_err_days": 999.0,
            "peak_magnitude_err_pct": 999.0,
            "water_balance_err_pct": 999.0,
        }

    o = q_obs[valid].astype(np.float64)
    s = np.clip(q_sim[valid].astype(np.float64), 0.0, None)  # Streamflow cannot be negative physically

    # 1. Nash-Sutcliffe Efficiency (NSE)
    obs_mean = np.mean(o)
    obs_var = np.sum((o - obs_mean) ** 2)
    res_var = np.sum((o - s) ** 2)
    nse = 1.0 - (res_var / obs_var) if obs_var > 1e-9 else -999.0

    # 2. Kling-Gupta Efficiency (KGE, Gupta et al. 2009)
    # KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
    s_std = np.std(s)
    o_std = np.std(o)
    s_mean = np.mean(s)

    if o_std > 1e-9 and s_std > 1e-9:
        r = float(np.corrcoef(o, s)[0, 1])
        if np.isnan(r):
            r = 0.0
        alpha = s_std / o_std
        beta = s_mean / obs_mean if obs_mean > 1e-9 else 1.0
        kge = 1.0 - float(np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    else:
        kge = -999.0

    # 3. Log-NSE (sensitive to low flows)
    log_o = np.log(o + epsilon)
    log_s = np.log(s + epsilon)
    log_obs_mean = np.mean(log_o)
    log_obs_var = np.sum((log_o - log_obs_mean) ** 2)
    log_res_var = np.sum((log_o - log_s) ** 2)
    log_nse = 1.0 - (log_res_var / log_obs_var) if log_obs_var > 1e-9 else -999.0

    # 4. RMSE
    rmse = float(np.sqrt(np.mean((s - o) ** 2)))

    # 5. MAE
    mae = float(np.mean(np.abs(s - o)))

    # 6. FHV: High-flow volume bias (top 2% exceedance, 98th percentile of observed)
    q98 = np.percentile(o, 98.0)
    high_mask = o >= q98
    if np.sum(high_mask) > 0 and np.sum(o[high_mask]) > 1e-6:
        fhv = float((np.sum(s[high_mask]) - np.sum(o[high_mask])) / np.sum(o[high_mask]) * 100.0)
    else:
        fhv = 0.0

    # 7. FLV: Low-flow volume bias (bottom 30% exceedance)
    q30 = np.percentile(o, 30.0)
    low_mask = o <= q30
    if np.sum(low_mask) > 0 and np.sum(o[low_mask]) > 1e-6:
        flv = float((np.sum(s[low_mask]) - np.sum(o[low_mask])) / np.sum(o[low_mask]) * 100.0)
    else:
        flv = 0.0

    # 8. Peak magnitude error (%)
    o_peak = float(np.max(o))
    s_peak = float(np.max(s))
    peak_mag_err = float((s_peak - o_peak) / o_peak * 100.0) if o_peak > 1e-6 else 0.0

    # 9. Peak timing error (days)
    o_peak_idx = int(np.argmax(o))
    s_peak_idx = int(np.argmax(s))
    peak_timing_err = float(abs(s_peak_idx - o_peak_idx))

    # 10. Water-balance error (%)
    tot_o = float(np.sum(o))
    tot_s = float(np.sum(s))
    wb_err = float((tot_s - tot_o) / tot_o * 100.0) if tot_o > 1e-6 else 0.0

    return {
        "nse": round(float(nse), 4),
        "kge": round(float(kge), 4),
        "log_nse": round(float(log_nse), 4),
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "fhv_pct": round(float(fhv), 2),
        "flv_pct": round(float(flv), 2),
        "peak_magnitude_err_pct": round(float(peak_mag_err), 2),
        "peak_timing_err_days": round(float(peak_timing_err), 1),
        "water_balance_err_pct": round(float(wb_err), 2),
    }
