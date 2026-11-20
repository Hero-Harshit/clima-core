"""SmokePath V2: Satellite Fire Radiative Power (FRP) & Biomass Burning Smoke Transport Engine.
Computes non-causal atmospheric advection indices, wind vector decomposition,
and CAMS AOD coupling for regional smoke transport across India.

MANDATORY SCIENTIFIC DISCLAIMER:
    This engine computes observational atmospheric transport indicators.
    It does NOT claim absolute legal source apportionment or causal attribution
    without continuous chemical mass balance (CMB), positive matrix factorization (PMF),
    or isotopic carbon-14 (14C) speciation data.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


# Major agricultural biomass burning centroid in NW India (Punjab / Haryana / Upper IGP)
FIRE_CENTROID_NW_LAT = 30.5
FIRE_CENTROID_NW_LON = 75.5

# Central India / Vidarbha / MP forest fire centroid
FIRE_CENTROID_CENTRAL_LAT = 21.5
FIRE_CENTROID_CENTRAL_LON = 79.0


def compute_haversine_distance_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    """Computes great-circle distance in km between arrays of points and a reference point."""
    r = 6371.0  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r * c


def compute_bearing_degrees(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Computes initial bearing from source (lat1, lon1) to receptor (lat2, lon2) in degrees [0, 360)."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlam = np.radians(lon2 - lon1)
    x = np.sin(dlam) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360.0) % 360.0


def compute_seasonal_fire_radiative_power(doy: np.ndarray) -> np.ndarray:
    """Models seasonal biomass burning FRP intensity (MW) across northern agricultural plains.
    Primary peak: Kharif paddy residue burning (Oct 15 - Nov 30, DOY ~288 to ~334).
    Secondary peak: Rabi wheat residue burning (Apr 15 - May 15, DOY ~105 to ~135).
    """
    # Gaussian peaks
    kharif_peak = 12000.0 * np.exp(-0.5 * ((doy - 305.0) / 14.0) ** 2)
    rabi_peak = 4500.0 * np.exp(-0.5 * ((doy - 120.0) / 12.0) ** 2)
    baseline = 250.0  # background domestic/municipal fires
    return kharif_peak + rabi_peak + baseline


class SmokePathTransportEngine:
    """Computes directional biomass burning smoke transport indices."""

    @staticmethod
    def enrich_smoke_features(df: pd.DataFrame) -> pd.DataFrame:
        """Adds 6 physical atmospheric advection and coupling features to dataset."""
        ws = df["wind_speed_10m_max"].values.astype(np.float64)
        wd = df["wind_direction_10m_dominant"].values.astype(np.float64)
        lats = df["latitude"].values.astype(np.float64)
        lons = df["longitude"].values.astype(np.float64)
        doy = df["day_of_year"].values.astype(np.float64)
        
        # 1. Wind vector components (meteorological convention: direction wind is blowing FROM)
        # Vector components u (eastward) and v (northward)
        rad_wd = np.radians(wd)
        u_wind = -ws * np.sin(rad_wd)
        v_wind = -ws * np.cos(rad_wd)
        
        # 2. Distance and bearing from NW burning centroid to each city
        dist_km = compute_haversine_distance_km(lats, lons, FIRE_CENTROID_NW_LAT, FIRE_CENTROID_NW_LON)
        bearing_to_city = compute_bearing_degrees(FIRE_CENTROID_NW_LAT, FIRE_CENTROID_NW_LON, lats, lons)
        
        # 3. Wind alignment factor: is wind blowing FROM the fire TOWARD the city?
        # Wind is blowing TOWARD (wd + 180) % 360
        wind_towards_deg = (wd + 180.0) % 360.0
        angle_diff_rad = np.radians(wind_towards_deg - bearing_to_city)
        alignment = np.maximum(0.0, np.cos(angle_diff_rad))
        
        # 4. Seasonal fire radiative power
        frp_est = compute_seasonal_fire_radiative_power(doy)
        
        # 5. Advection index T_smoke = (FRP / (d^2)) * alignment * (ws / 5) * 1e4
        # Decays with inverse square of distance, scaled by wind alignment and speed
        smoke_advection = (frp_est / ((dist_km + 40.0) ** 1.8)) * alignment * np.clip(ws / 4.0, 0.2, 3.0) * 100.0
        
        # 6. AOD coupling and ventilation index
        aod = df["aod_mean"].values.astype(np.float64) if "aod_mean" in df.columns else np.zeros_like(ws)
        aod_coupling = smoke_advection * np.maximum(aod, 0.05)
        
        # Ventilation proxy: wind speed * diurnal temperature range
        t_max = df["temperature_2m_max"].values.astype(np.float64)
        t_min = df["temperature_2m_min"].values.astype(np.float64)
        dtr = np.maximum(t_max - t_min, 1.0)
        ventilation_proxy = ws * dtr
        
        # Winter boundary layer stagnation index: RH / max(ws, 0.5)
        rh = df["relative_humidity_2m_mean"].values.astype(np.float64)
        stagnation_index = rh / np.maximum(ws, 0.5)
        
        df_out = df.copy()
        df_out["wind_u_component"] = u_wind.astype(np.float32)
        df_out["wind_v_component"] = v_wind.astype(np.float32)
        df_out["smoke_advection_index"] = smoke_advection.astype(np.float32)
        df_out["aod_smoke_coupling"] = aod_coupling.astype(np.float32)
        df_out["ventilation_index_proxy"] = ventilation_proxy.astype(np.float32)
        df_out["stagnation_inversion_index"] = stagnation_index.astype(np.float32)
        
        return df_out
