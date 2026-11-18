"""
Elevation and Pressure-Aware Psychrometric Thermodynamics.
Computes station barometric pressure and psychrometric wet-bulb temperature,
correcting sea-level assumptions for elevated Indian urban plateaus
(e.g., Bengaluru 920m, Pune 560m, Hyderabad 540m).
"""

from __future__ import annotations

import math
from typing import Dict, Any, Optional
import numpy as np

# Standard elevation in meters for the 50 major Indian cities
CITY_ELEVATIONS_M: Dict[str, float] = {
    "delhi": 214.0,
    "mumbai": 6.0,
    "kolkata": 12.0,
    "bengaluru": 920.0,
    "bangalore": 920.0,
    "chennai": 6.0,
    "hyderabad": 542.0,
    "ahmedabad": 53.0,
    "pune": 560.0,
    "surat": 13.0,
    "jaipur": 431.0,
    "lucknow": 123.0,
    "kanpur": 126.0,
    "nagpur": 310.0,
    "indore": 553.0,
    "thane": 7.0,
    "bhopal": 527.0,
    "visakhapatnam": 45.0,
    "patna": 53.0,
    "vadodara": 39.0,
    "ghaziabad": 214.0,
    "ludhiana": 244.0,
    "agra": 171.0,
    "nashik": 600.0,
    "faridabad": 198.0,
    "meerut": 219.0,
    "rajkot": 128.0,
    "varanasi": 81.0,
    "srinagar": 1585.0,
    "aurangabad": 569.0,
    "dhanbad": 227.0,
    "amritsar": 234.0,
    "navi mumbai": 10.0,
    "allahabad": 98.0,
    "prayagraj": 98.0,
    "ranchi": 651.0,
    "howrah": 12.0,
    "coimbatore": 411.0,
    "jabalpur": 411.0,
    "gwalior": 197.0,
    "vijayawada": 11.0,
    "jodhpur": 231.0,
    "madurai": 101.0,
    "raipur": 298.0,
    "kota": 271.0,
    "chandigarh": 321.0,
    "guwahati": 55.0,
    "solapur": 458.0,
    "hubli": 671.0,
    "bareilly": 166.0,
    "moradabad": 198.0,
    "mysore": 763.0,
    "gurgaon": 219.0,
    "noida": 200.0,
    "shimla": 2206.0,
    "dehradun": 640.0,
}


def estimate_station_pressure_hpa(elevation_m: float) -> float:
    """
    Calculate standard barometric station pressure using the international barometric formula.
    P = P0 * (1 - 2.25577e-5 * z) ^ 5.25588
    """
    z = max(0.0, float(elevation_m))
    p = 1013.25 * math.pow(1.0 - (2.25577e-5 * z), 5.25588)
    return round(p, 1)


def get_city_elevation(city: str) -> float:
    """Lookup city elevation in meters, default to 150m if unknown."""
    return CITY_ELEVATIONS_M.get(city.strip().lower(), 150.0)


def saturation_vapor_pressure_hpa(temp_c: float) -> float:
    """
    Calculate saturation vapor pressure over liquid water (Magnus-Tetens formula).
    Valid from -40°C to +50°C.
    """
    t = float(temp_c)
    return 6.1078 * math.exp((17.27 * t) / (t + 237.3))


def estimate_stull_wet_bulb(temp_c: float, rh_pct: float) -> float:
    """
    Stull (2011) empirical psychrometric wet-bulb formula at sea level (1013.25 hPa).
    Mean absolute error < 0.3°C across standard meteorological ranges.
    """
    t = float(temp_c)
    rh = float(np.clip(rh_pct, 1.0, 99.0))
    tw = (
        t * math.atan(0.151977 * math.sqrt(rh + 8.313659))
        + math.atan(t + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * math.pow(rh, 1.5) * math.atan(0.023101 * rh)
        - 4.686035
    )
    return round(tw, 2)


def calculate_elevation_aware_wet_bulb(
    temp_c: float,
    rh_pct: float,
    city: Optional[str] = None,
    elevation_m: Optional[float] = None,
    pressure_hpa: Optional[float] = None
) -> Dict[str, Any]:
    """
    Compute pressure-aware psychrometric wet-bulb temperature.
    Solves the exact psychrometric heat balance:
      e = e_s(Tw) - gamma(P) * (T - Tw)
    where gamma(P) = 0.000665 * P is the psychrometric constant.

    At higher elevations (lower pressure), evaporative cooling is enhanced,
    producing a cooler wet-bulb temperature than the sea-level Stull approximation.
    """
    t = float(temp_c)
    rh = float(np.clip(rh_pct, 1.0, 99.0))

    if pressure_hpa is not None:
        p_station = float(pressure_hpa)
        elev_m = elevation_m or 0.0
    elif elevation_m is not None:
        elev_m = float(elevation_m)
        p_station = estimate_station_pressure_hpa(elev_m)
    elif city is not None:
        elev_m = get_city_elevation(city)
        p_station = estimate_station_pressure_hpa(elev_m)
    else:
        elev_m = 0.0
        p_station = 1013.25

    # 1. Sea-level Stull baseline
    tw_stull_sea_level = estimate_stull_wet_bulb(t, rh)

    # 2. Exact physical psychrometric solver
    # e: actual vapor pressure (hPa)
    es_dry = saturation_vapor_pressure_hpa(t)
    e_actual = es_dry * (rh / 100.0)

    # Psychrometric constant gamma = Cp * P / (epsilon * lambda) ~ 0.000665 * P
    gamma = 0.000665 * p_station

    # Objective function: f(Tw) = e_s(Tw) - gamma * (T - Tw) - e_actual = 0
    # Tw must be in [DewPoint, DryBulb]
    # Simple bisection solver (converges in ~12 iterations to < 0.01°C)
    low = -15.0
    high = t + 0.1

    for _ in range(16):
        mid = (low + high) / 2.0
        mid_es = saturation_vapor_pressure_hpa(mid)
        val = mid_es - (gamma * (t - mid)) - e_actual
        if val < 0:
            low = mid
        else:
            high = mid

    tw_exact = round(float((low + high) / 2.0), 2)
    pressure_delta = round(tw_exact - tw_stull_sea_level, 2)

    return {
        "wet_bulb_temp_c": tw_exact,
        "sea_level_stull_wet_bulb_c": tw_stull_sea_level,
        "elevation_m": elev_m,
        "station_pressure_hpa": p_station,
        "elevation_cooling_delta_c": pressure_delta,
        "dry_bulb_temp_c": t,
        "relative_humidity_pct": rh,
    }


if __name__ == "__main__":
    for c in ["Mumbai", "Delhi", "Pune", "Bengaluru", "Shimla"]:
        res = calculate_elevation_aware_wet_bulb(temp_c=38.0, rh_pct=60.0, city=c)
        print(f"{c:12} (Elev: {res['elevation_m']:4.0f}m, P: {res['station_pressure_hpa']} hPa) -> "
              f"Wet-Bulb: {res['wet_bulb_temp_c']}°C (Sea-level Stull: {res['sea_level_stull_wet_bulb_c']}°C, Delta: {res['elevation_cooling_delta_c']:+.2f}°C)")
