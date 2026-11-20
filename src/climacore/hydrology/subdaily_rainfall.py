"""Sub-Daily Rainfall Disaggregation and Pluvial Flash-Flood Screening Module.
Provides scientifically grounded temporal rainfall disaggregation:

1. Huff Temporal Storm Distribution Curves (Quartiles 1, 2, 3, 4) (Huff, 1967).
2. Method of Fragments for 24-hour daily-to-hourly disaggregation (Srikanthan & McMahon, 2001).
3. Strict daily volume preservation: sum(h_i) == P_daily.
4. Strict data provenance tagging (OBSERVED, PREDICTED, CALCULATED, SCENARIO).
5. Urban pluvial flash-flood hazard screening index with honest engineering domain boundaries.

References:
  - Huff, F. A. (1967). Time distribution of rainfall in heavy storms. Water Resources Research, 3(4), 1007-1019.
  - Srikanthan, R., & McMahon, T. A. (2001). Stochastic generation of annual, monthly and daily climate data:
    a review. Hydrology and Earth System Sciences, 5(4), 653-670.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


class DataProvenance(str, Enum):
    """Authoritative data provenance classification for meteorological and hydrological series."""
    OBSERVED = "OBSERVED"        # Direct physical sensor or ground telemetry (IMD AWS/ARG)
    PREDICTED = "PREDICTED"      # Supervised machine learning or NWP model forecast output
    CALCULATED = "CALCULATED"    # Deterministic physical formulation (e.g. SCS-CN volume)
    SCENARIO = "SCENARIO"        # Synthetic or disaggregated profile (Huff curves, fragments)


# Normalized Huff Cumulative Distribution Curves (5% increments in storm duration: 0.0 to 1.0)
# Represents 50% median probability ordinate for heavy convective storms
HUFF_PERCENTILES = np.linspace(0.0, 1.0, 21)

HUFF_CURVES: Dict[str, np.ndarray] = {
    # Q1: Convective cloudburst (burst in first 25% of duration)
    "Q1": np.array([
        0.00, 0.16, 0.33, 0.50, 0.63, 0.71, 0.76, 0.81, 0.84, 0.87,
        0.89, 0.91, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.00
    ]),
    # Q2: Typical moderate convective front (peak between 25% and 50%)
    "Q2": np.array([
        0.00, 0.03, 0.08, 0.14, 0.23, 0.38, 0.54, 0.65, 0.73, 0.79,
        0.84, 0.88, 0.91, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.00
    ]),
    # Q3: Extended afternoon frontal storm (peak between 50% and 75%)
    "Q3": np.array([
        0.00, 0.02, 0.05, 0.08, 0.12, 0.16, 0.22, 0.29, 0.39, 0.54,
        0.68, 0.79, 0.86, 0.90, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00
    ]),
    # Q4: Decaying tropical depression (peak in final 25% of duration)
    "Q4": np.array([
        0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.19, 0.23,
        0.28, 0.34, 0.42, 0.52, 0.65, 0.78, 0.87, 0.93, 0.97, 0.99, 1.00
    ]),
}


def disaggregate_daily_huff(
    daily_rainfall_mm: float,
    quartile: str = "Q1",
    n_hours: int = 24,
    provenance: DataProvenance = DataProvenance.SCENARIO,
) -> Dict[str, Any]:
    """Disaggregates daily rainfall into hourly intervals using Huff temporal distribution curves.
    
    Parameters:
      daily_rainfall_mm: Total 24h precipitation in mm (must be non-negative).
      quartile: Huff storm quartile ('Q1', 'Q2', 'Q3', or 'Q4').
      n_hours: Number of hourly intervals (default: 24).
      provenance: Origin tag for the output series (default: SCENARIO).
      
    Returns:
      Dictionary containing hourly rainfall array, cumulative curve, and volume check.
    """
    daily_p = max(0.0, float(daily_rainfall_mm))
    if quartile not in HUFF_CURVES:
        raise ValueError(f"Invalid Huff quartile: {quartile}. Must be one of {list(HUFF_CURVES.keys())}.")

    if daily_p == 0.0:
        return {
            "daily_total_mm": 0.0,
            "disaggregated_total_mm": 0.0,
            "hourly_rainfall_mm": [0.0] * n_hours,
            "cumulative_rainfall_mm": [0.0] * n_hours,
            "volume_closure_error_mm": 0.0,
            "quartile": quartile,
            "provenance": provenance.value,
        }

    # Interpolate Huff cumulative fractions to hourly resolution (0 to 1 over n_hours)
    t_eval = np.linspace(0.0, 1.0, n_hours + 1)
    cum_fractions = np.interp(t_eval, HUFF_PERCENTILES, HUFF_CURVES[quartile])
    
    # Incremental hourly depths
    hourly_depths = np.diff(cum_fractions) * daily_p
    
    # Strict mass-balance normalization: ensure sum(hourly) == daily_p to exact machine float
    sum_raw = np.sum(hourly_depths)
    if sum_raw > 0:
        hourly_depths = hourly_depths * (daily_p / sum_raw)
    
    cum_depths = np.cumsum(hourly_depths)
    closure_err = abs(daily_p - float(np.sum(hourly_depths)))

    return {
        "daily_total_mm": round(daily_p, 4),
        "disaggregated_total_mm": round(float(np.sum(hourly_depths)), 4),
        "hourly_rainfall_mm": [round(float(h), 4) for h in hourly_depths],
        "cumulative_rainfall_mm": [round(float(c), 4) for c in cum_depths],
        "volume_closure_error_mm": round(closure_err, 8),
        "peak_hour_index": int(np.argmax(hourly_depths)),
        "peak_1h_mm": round(float(np.max(hourly_depths)), 4),
        "quartile": quartile,
        "provenance": provenance.value,
    }


def disaggregate_method_of_fragments(
    daily_rainfall_mm: float,
    fragment_weights: Optional[Union[List[float], np.ndarray]] = None,
    provenance: DataProvenance = DataProvenance.SCENARIO,
) -> Dict[str, Any]:
    """Disaggregates daily rainfall using an observed or standardized vector of hourly fragment weights.
    
    Parameters:
      daily_rainfall_mm: Total 24h precipitation in mm.
      fragment_weights: Array of 24 positive hourly fractions summing to 1.0.
      provenance: Origin tag for the output series.
      
    Returns:
      Dictionary with hourly rainfall, closure error, and provenance.
    """
    daily_p = max(0.0, float(daily_rainfall_mm))

    if fragment_weights is None:
        # Default empirical coastal monsoon diurnal profile (peak during early morning 03:00-07:00)
        default_weights = np.array([
            0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.11, 0.08, 0.05, 0.04,
            0.03, 0.02, 0.02, 0.02, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02,
            0.02, 0.01, 0.01, 0.01
        ], dtype=np.float64)
        fragment_weights = default_weights

    weights = np.asarray(fragment_weights, dtype=np.float64)
    if len(weights) != 24:
        raise ValueError(f"Fragment weights must have length 24, got {len(weights)}.")

    # Normalize weights to exactly 1.0
    w_sum = np.sum(weights)
    if w_sum <= 0:
        norm_weights = np.ones(24, dtype=np.float64) / 24.0
    else:
        norm_weights = weights / w_sum

    hourly_depths = norm_weights * daily_p
    closure_err = abs(daily_p - float(np.sum(hourly_depths)))

    return {
        "daily_total_mm": round(daily_p, 4),
        "disaggregated_total_mm": round(float(np.sum(hourly_depths)), 4),
        "hourly_rainfall_mm": [round(float(h), 4) for h in hourly_depths],
        "volume_closure_error_mm": round(closure_err, 8),
        "peak_hour_index": int(np.argmax(hourly_depths)),
        "peak_1h_mm": round(float(np.max(hourly_depths)), 4),
        "provenance": provenance.value,
    }


def screen_pluvial_flash_flood(
    hourly_rainfall_mm: List[float],
    soil_moisture_index: float = 0.5,
    impervious_fraction: float = 0.65,
    is_tidal_spring_high: bool = False,
    city_name: str = "Mumbai",
) -> Dict[str, Any]:
    """Evaluates urban pluvial flash-flood potential from sub-daily rainfall bursts and local boundary conditions.
    
    Parameters:
      hourly_rainfall_mm: 24-hour sequence of hourly rainfall depths.
      soil_moisture_index: Normalized antecedent topsoil moisture saturation (0.0 to 1.0).
      impervious_fraction: Urban core connected impervious surface fraction (0.0 to 1.0).
      is_tidal_spring_high: True if coastal storm surge / spring high tide coincides with downpour.
      city_name: Name of the pilot city.
      
    Returns:
      Comprehensive flash-flood hazard score, warning category, and physical limiting factors.
    """
    h_arr = np.asarray(hourly_rainfall_mm, dtype=np.float64)
    peak_1h = float(np.max(h_arr)) if len(h_arr) > 0 else 0.0
    
    # Calculate rolling 3-hour peak
    if len(h_arr) >= 3:
        h_series = np.convolve(h_arr, np.ones(3), mode="valid")
        peak_3h = float(np.max(h_series))
    else:
        peak_3h = float(np.sum(h_arr))

    # Base rainfall intensity index (0 to 50 points)
    # Heavy burst: >30 mm/h is severe; >65 mm/h is catastrophic cloudburst
    intensity_score = min(50.0, (peak_1h / 65.0) * 35.0 + (peak_3h / 120.0) * 15.0)

    # Antecedent saturation modifier (0 to 20 points)
    saturation_score = float(np.clip(soil_moisture_index * 20.0, 0.0, 20.0))

    # Urban imperviousness modifier (0 to 15 points)
    impervious_score = float(np.clip(impervious_fraction * 15.0, 0.0, 15.0))

    # Coastal tidal surge backwater penalty (0 or 15 points)
    tidal_score = 15.0 if is_tidal_spring_high else 0.0

    total_hazard_score = min(100.0, intensity_score + saturation_score + impervious_score + tidal_score)

    if total_hazard_score >= 75.0:
        category = "EMERGENCY"
        action = "Issue immediate Red flash-flood warning; deploy NDRF/SDRF pumps; close low-lying subways."
    elif total_hazard_score >= 50.0:
        category = "HIGH"
        action = "Issue Orange alert; clear storm nullah choke-points; alert municipal disaster cells."
    elif total_hazard_score >= 30.0:
        category = "MODERATE"
        action = "Issue Yellow advisory; monitor radar and tidal tables; inspect street drains."
    else:
        category = "LOW"
        action = "Normal routine stormwater surveillance."

    return {
        "pilot_city": city_name,
        "hazard_score": round(total_hazard_score, 2),
        "warning_category": category,
        "peak_1h_rainfall_mm": round(peak_1h, 2),
        "peak_3h_rainfall_mm": round(peak_3h, 2),
        "contributing_factors": {
            "rainfall_intensity_points": round(intensity_score, 2),
            "soil_saturation_points": round(saturation_score, 2),
            "urban_imperviousness_points": round(impervious_score, 2),
            "tidal_lock_points": round(tidal_score, 2),
        },
        "coastal_tidal_lock_active": is_tidal_spring_high,
        "recommended_action": action,
        "engineering_boundary_limitations": (
            "Screening index identifies surface pluvial inundation potential and tidal backwater stalling. "
            "Does NOT model subterranean pipe discharge without municipal engineering conduit dimensions and invert elevations."
        ),
        "provenance": DataProvenance.CALCULATED.value,
    }
