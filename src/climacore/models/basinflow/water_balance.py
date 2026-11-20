"""Water Balance Accounting and Hydrograph Separation Module.
Provides scientifically rigorous water balance closure and hydrograph baseflow/quickflow
separation following the Eckhardt (2005) two-parameter recursive digital filter.

References:
  - Eckhardt, K. (2005). How to construct recursive digital filters for baseflow separation.
    Hydrological Processes, 19(2), 507-515.
  - Lyne, V., & Hollick, M. (1979). Stochastic time-variable rainfall-runoff modelling.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


def eckhardt_filter(
    q: np.ndarray,
    alpha: float = 0.98,
    bfi_max: float = 0.80,
) -> Tuple[np.ndarray, np.ndarray]:
    """Separates streamflow into baseflow (slowflow) and quickflow using the Eckhardt (2005) filter.
    
    Parameters:
      q: Total observed or simulated streamflow (mm/day or m3/s). Must be non-negative.
      alpha: Filter parameter (recession constant), typically 0.95 - 0.99 for daily data.
      bfi_max: Maximum Baseflow Index, typically 0.80 for perennial streams, 0.50 for ephemeral.
      
    Returns:
      (baseflow, quickflow) arrays with identical shape and non-negativity guarantee.
    """
    n = len(q)
    baseflow = np.zeros(n, dtype=np.float64)
    quickflow = np.zeros(n, dtype=np.float64)
    
    if n == 0:
        return baseflow, quickflow
        
    # Initial baseflow condition
    first_valid = 0.0
    for val in q:
        if not np.isnan(val) and val >= 0.0:
            first_valid = float(val) * bfi_max
            break
            
    b_prev = first_valid
    denom = 1.0 - alpha * bfi_max
    
    for t in range(n):
        qt = float(q[t])
        if np.isnan(qt) or qt < 0.0:
            baseflow[t] = np.nan
            quickflow[t] = np.nan
            continue
            
        # Eckhardt recursive formula
        num = (1.0 - bfi_max) * alpha * b_prev + (1.0 - alpha) * bfi_max * qt
        bt = num / denom
        
        # Physical constraints: 0 <= bt <= qt
        bt = max(0.0, min(bt, qt))
        
        baseflow[t] = bt
        quickflow[t] = max(0.0, qt - bt)
        b_prev = bt
        
    return baseflow, quickflow


def compute_water_balance(
    prcp: np.ndarray,
    streamflow: np.ndarray,
    pet: Optional[np.ndarray] = None,
    catchment_area_km2: Optional[float] = None,
    alpha: float = 0.98,
    bfi_max: float = 0.80,
) -> Dict[str, Any]:
    """Computes comprehensive water balance accounting, runoff coefficient, and hydrograph partitioning.
    
    Parameters:
      prcp: Daily precipitation series (mm/day).
      streamflow: Daily streamflow series (mm/day).
      pet: Optional daily potential evapotranspiration (mm/day).
      catchment_area_km2: Optional catchment drainage area for volumetric calculation (km2).
      alpha: Eckhardt filter recession parameter (default: 0.98).
      bfi_max: Maximum baseflow index (default: 0.80).
      
    Returns:
      Dictionary of water balance metrics and component totals.
    """
    prcp = np.asarray(prcp, dtype=np.float64)
    streamflow = np.asarray(streamflow, dtype=np.float64)
    
    # Filter common valid masks
    valid_mask = (~np.isnan(prcp)) & (~np.isnan(streamflow)) & (prcp >= 0.0) & (streamflow >= 0.0)
    p_clean = prcp[valid_mask]
    q_clean = streamflow[valid_mask]
    
    if len(p_clean) == 0:
        return {
            "error": "No overlapping valid daily records found",
            "total_days": 0,
        }
        
    p_total_mm = float(np.sum(p_clean))
    q_total_mm = float(np.sum(q_clean))
    
    # Runoff coefficient
    runoff_coefficient = (q_total_mm / p_total_mm) if p_total_mm > 0.0 else 0.0
    
    # Baseflow separation on clean streamflow
    baseflow_clean, quickflow_clean = eckhardt_filter(q_clean, alpha=alpha, bfi_max=bfi_max)
    
    q_quick_mm = float(np.sum(quickflow_clean))
    q_slow_mm = float(np.sum(baseflow_clean))
    
    # Proportions of Streamflow Q
    quickflow_pct_of_q = (q_quick_mm / q_total_mm * 100.0) if q_total_mm > 0.0 else 0.0
    slowflow_pct_of_q = (q_slow_mm / q_total_mm * 100.0) if q_total_mm > 0.0 else 0.0
    
    # Proportions of Precipitation P
    quickflow_pct_of_p = (q_quick_mm / p_total_mm * 100.0) if p_total_mm > 0.0 else 0.0
    slowflow_pct_of_p = (q_slow_mm / p_total_mm * 100.0) if p_total_mm > 0.0 else 0.0
    
    # Evapotranspiration + Deep Storage Change (P - Q)
    loss_and_storage_mm = max(0.0, p_total_mm - q_total_mm)
    loss_and_storage_pct_of_p = (loss_and_storage_mm / p_total_mm * 100.0) if p_total_mm > 0.0 else 0.0
    
    # Closure verification: Q_quick + Q_slow should equal Q exactly
    hydrograph_closure_error_mm = abs(q_total_mm - (q_quick_mm + q_slow_mm))
    mass_balance_closure_ratio = ((q_total_mm + loss_and_storage_mm) / p_total_mm) if p_total_mm > 0.0 else 1.0
    
    result: Dict[str, Any] = {
        "valid_evaluation_days": int(len(p_clean)),
        "precipitation": {
            "total_mm": round(p_total_mm, 2),
            "mean_daily_mm": round(float(np.mean(p_clean)), 2),
            "max_daily_mm": round(float(np.max(p_clean)), 2),
        },
        "streamflow": {
            "total_mm": round(q_total_mm, 2),
            "mean_daily_mm": round(float(np.mean(q_clean)), 2),
            "max_daily_mm": round(float(np.max(q_clean)), 2),
            "runoff_coefficient": round(runoff_coefficient, 4),
            "runoff_coefficient_pct": round(runoff_coefficient * 100.0, 2),
        },
        "hydrograph_partitioning": {
            "quickflow_total_mm": round(q_quick_mm, 2),
            "quickflow_pct_of_streamflow": round(quickflow_pct_of_q, 2),
            "quickflow_pct_of_precipitation": round(quickflow_pct_of_p, 2),
            "slowflow_total_mm": round(q_slow_mm, 2),
            "slowflow_pct_of_streamflow": round(slowflow_pct_of_q, 2),
            "slowflow_pct_of_precipitation": round(slowflow_pct_of_p, 2),
            "hydrograph_closure_error_mm": round(hydrograph_closure_error_mm, 6),
        },
        "et_and_storage": {
            "et_plus_storage_mm": round(loss_and_storage_mm, 2),
            "et_plus_storage_pct_of_p": round(loss_and_storage_pct_of_p, 2),
            "mass_balance_closure_ratio": round(mass_balance_closure_ratio, 4),
        },
        "provenance_notes": (
            "Quickflow and Slowflow percentages are reported relative to total streamflow Q. "
            "Residual (P - Q) corresponds to actual evapotranspiration and subsurface storage changes."
        ),
    }
    
    if catchment_area_km2 is not None and catchment_area_km2 > 0:
        area_m2 = catchment_area_km2 * 1_000_000.0
        # 1 mm = 0.001 m depth -> volume = depth * area
        p_vol_m3 = (p_total_mm / 1000.0) * area_m2
        q_vol_m3 = (q_total_mm / 1000.0) * area_m2
        result["volumetric_m3"] = {
            "catchment_area_km2": float(catchment_area_km2),
            "precipitation_m3": round(p_vol_m3, 0),
            "streamflow_m3": round(q_vol_m3, 0),
            "quickflow_m3": round((q_quick_mm / 1000.0) * area_m2, 0),
            "slowflow_m3": round((q_slow_mm / 1000.0) * area_m2, 0),
        }
        
    return result
