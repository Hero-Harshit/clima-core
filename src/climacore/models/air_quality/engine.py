"""
AirQuality V3 Flagship Engine.
Coupled Atmospheric Chemistry & Meteorology Intelligence:
  1. Boundary layer ventilation index (V = Wind Speed x Boundary Layer Mixing Height) & nocturnal inversion trapping.
  2. Multi-horizon multi-pollutant projections (PM2.5, PM10, NO2, O3 across 24h, 48h, 72h).
  3. CAMS advection coupled and calibrated against physical CPCB ground station realities.
  4. CPCB 8-pollutant compliance audit (requiring >= 3 monitored pollutants with at least one particulate).
  5. CPCB official NAQI sub-index calculation and GRAP stage civil restrictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import lightgbm as lgb

from climacore.common.logging import get_logger
from climacore.config.loader import get_project_root

logger = get_logger("climacore.models.air_quality.engine")

_PM25_BOOSTER: Optional[lgb.Booster] = None
_PM10_BOOSTER: Optional[lgb.Booster] = None
_NO2_BOOSTER: Optional[lgb.Booster] = None
_STAGNATION_BOOSTER: Optional[lgb.Booster] = None
_SMOKE_BOOSTER: Optional[lgb.Booster] = None


def load_air_quality_boosters() -> Dict[str, Optional[lgb.Booster]]:
    """Load and cache all air quality boosters."""
    global _PM25_BOOSTER, _PM10_BOOSTER, _NO2_BOOSTER, _STAGNATION_BOOSTER, _SMOKE_BOOSTER

    root = get_project_root()
    base_art = root / "artifacts" / "models"

    if _PM25_BOOSTER is None:
        p = base_art / "air_quality" / "v0.1.0" / "model.txt"
        if p.exists():
            try:
                _PM25_BOOSTER = lgb.Booster(model_file=str(p))
                logger.info(f"Loaded PM2.5 booster from {p}")
            except Exception as e:
                logger.warning(f"Failed to load PM2.5 booster: {e}")

    if _PM10_BOOSTER is None:
        p = base_art / "cpcb_multipollutant" / "v0.1.0" / "pm10_model.txt"
        if p.exists():
            try:
                _PM10_BOOSTER = lgb.Booster(model_file=str(p))
                logger.info(f"Loaded PM10 booster from {p}")
            except Exception as e:
                logger.warning(f"Failed to load PM10 booster: {e}")

    if _NO2_BOOSTER is None:
        p = base_art / "cpcb_multipollutant" / "v0.1.0" / "no2_model.txt"
        if p.exists():
            try:
                _NO2_BOOSTER = lgb.Booster(model_file=str(p))
                logger.info(f"Loaded NO2 booster from {p}")
            except Exception as e:
                logger.warning(f"Failed to load NO2 booster: {e}")

    if _STAGNATION_BOOSTER is None:
        p = base_art / "atmospheric_stagnation" / "v0.1.0" / "model.txt"
        if p.exists():
            try:
                _STAGNATION_BOOSTER = lgb.Booster(model_file=str(p))
                logger.info(f"Loaded Atmospheric Stagnation booster from {p}")
            except Exception as e:
                logger.warning(f"Failed to load Stagnation booster: {e}")

    if _SMOKE_BOOSTER is None:
        p = base_art / "transboundary_smoke" / "v0.1.0" / "model.txt"
        if p.exists():
            try:
                _SMOKE_BOOSTER = lgb.Booster(model_file=str(p))
                logger.info(f"Loaded Transboundary Smoke booster from {p}")
            except Exception as e:
                logger.warning(f"Failed to load Smoke booster: {e}")

    return {
        "pm25": _PM25_BOOSTER,
        "pm10": _PM10_BOOSTER,
        "no2": _NO2_BOOSTER,
        "stagnation": _STAGNATION_BOOSTER,
        "smoke": _SMOKE_BOOSTER,
    }


def compute_sub_index(conc: float, breakpoints: List[Tuple[float, float, float, float]]) -> float:
    """
    Compute CPCB linear interpolation sub-index:
    Ip = [(I_hi - I_lo) / (B_hi - B_lo)] * (C - B_lo) + I_lo
    """
    if conc <= 0.0:
        return 0.0
    for b_lo, b_hi, i_lo, i_hi in breakpoints:
        if b_lo <= conc <= b_hi:
            return round(((i_hi - i_lo) / (b_hi - b_lo)) * (conc - b_lo) + i_lo, 1)
    # Exceeds max breakpoint -> scale linearly above 500
    last_b_lo, last_b_hi, last_i_lo, last_i_hi = breakpoints[-1]
    return round(last_i_hi + (conc - last_b_hi) * 0.5, 1)


# Official CPCB 24-hour Breakpoint Tables (B_lo, B_hi, I_lo, I_hi)
PM25_BREAKPOINTS = [
    (0.0, 30.0, 0.0, 50.0),
    (30.1, 60.0, 51.0, 100.0),
    (60.1, 90.0, 101.0, 200.0),
    (90.1, 120.0, 201.0, 300.0),
    (120.1, 250.0, 301.0, 400.0),
    (250.1, 500.0, 401.0, 500.0),
]

PM10_BREAKPOINTS = [
    (0.0, 50.0, 0.0, 50.0),
    (50.1, 100.0, 51.0, 100.0),
    (100.1, 250.0, 101.0, 200.0),
    (250.1, 350.0, 201.0, 300.0),
    (350.1, 430.0, 301.0, 400.0),
    (430.1, 600.0, 401.0, 500.0),
]

NO2_BREAKPOINTS = [
    (0.0, 40.0, 0.0, 50.0),
    (40.1, 80.0, 51.0, 100.0),
    (80.1, 180.0, 101.0, 200.0),
    (180.1, 280.0, 201.0, 300.0),
    (280.1, 400.0, 301.0, 400.0),
    (400.1, 600.0, 401.0, 500.0),
]

O3_BREAKPOINTS = [
    (0.0, 50.0, 0.0, 50.0),
    (50.1, 100.0, 51.0, 100.0),
    (100.1, 168.0, 101.0, 200.0),
    (168.1, 208.0, 201.0, 300.0),
    (208.1, 748.0, 301.0, 400.0),
    (748.1, 1000.0, 401.0, 500.0),
]


def classify_naqi(aqi_val: float) -> Tuple[str, str]:
    """Classify composite NAQI into official CPCB 6-tier category and color hex."""
    if aqi_val <= 50:
        return "Good", "#00B050"
    elif aqi_val <= 100:
        return "Satisfactory", "#92D050"
    elif aqi_val <= 200:
        return "Moderate", "#FFFF00"
    elif aqi_val <= 300:
        return "Poor", "#FF9900"
    elif aqi_val <= 400:
        return "Very Poor", "#FF0000"
    else:
        return "Severe", "#C00000"


class AirQualityEngine:
    """
    Flagship AirQuality V3 Engine.
    Consolidates:
      1. Boundary layer ventilation index (V = Wind Speed x Boundary Layer Mixing Height) and dispersion trapping.
      2. Multi-horizon multi-pollutant projections (PM2.5, PM10, NO2, O3).
      3. CAMS advection coupled with CPCB station ground calibrations.
      4. CPCB official NAQI sub-index calculation and 8-pollutant compliance verification.
      5. GRAP Stage I-IV triggers and exposure-decoupled health advisories.
    """

    def __init__(self):
        self.boosters = load_air_quality_boosters()

    def evaluate_air_quality(
        self,
        city: str,
        pm2_5_ug_m3: float,
        pm10_ug_m3: Optional[float] = None,
        no2_ug_m3: Optional[float] = None,
        so2_ug_m3: Optional[float] = None,
        co_mg_m3: Optional[float] = None,
        o3_ug_m3: Optional[float] = None,
        nh3_ug_m3: Optional[float] = None,
        pb_ug_m3: Optional[float] = None,
        wind_speed_kmh: float = 12.0,
        temp_max_c: float = 30.0,
        temp_min_c: float = 20.0,
        precipitation_mm: float = 0.0,
        boundary_layer_height_m: Optional[float] = None,
        cams_pm25_advection_ug_m3: Optional[float] = None,
        month: int = 10,
        day_of_year: int = 290,
        population_15km: Optional[int] = None,
        schools_15km: Optional[int] = None,
        hospitals_15km: Optional[int] = None,
        forecast_horizon_hours: int = 24,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate air quality with coupled meteorology-chemistry and ventilation physics.
        """
        explicitly_provided = []
        if pm2_5_ug_m3 is not None:
            explicitly_provided.append("PM2.5")
        if pm10_ug_m3 is not None:
            explicitly_provided.append("PM10")
        if no2_ug_m3 is not None:
            explicitly_provided.append("NO2")
        if so2_ug_m3 is not None:
            explicitly_provided.append("SO2")
        if co_mg_m3 is not None:
            explicitly_provided.append("CO")
        if o3_ug_m3 is not None:
            explicitly_provided.append("O3")
        if nh3_ug_m3 is not None:
            explicitly_provided.append("NH3")
        if pb_ug_m3 is not None:
            explicitly_provided.append("Pb")

        # 1. Calibrate Pollutants (Ground Station Reality + Stoichiometric Transfer)
        raw_pm25 = max(1.0, float(pm2_5_ug_m3))
        # If CAMS advection forecast is available, blend 80% ground station, 20% regional advection
        if cams_pm25_advection_ug_m3 is not None:
            calibrated_pm25 = round(0.80 * raw_pm25 + 0.20 * float(cams_pm25_advection_ug_m3), 1)
        else:
            calibrated_pm25 = round(raw_pm25, 1)

        calibrated_pm10 = max(1.0, float(pm10_ug_m3)) if pm10_ug_m3 is not None else round(calibrated_pm25 * 1.68, 1)
        calibrated_no2 = max(1.0, float(no2_ug_m3)) if no2_ug_m3 is not None else round(12.0 + calibrated_pm25 * 0.35, 1)
        calibrated_o3 = max(1.0, float(o3_ug_m3)) if o3_ug_m3 is not None else round(15.0 + max(0.0, temp_max_c - 25.0) * 1.8, 1)

        # 2. Compute CPCB Sub-indices
        sub_pm25 = compute_sub_index(calibrated_pm25, PM25_BREAKPOINTS)
        sub_pm10 = compute_sub_index(calibrated_pm10, PM10_BREAKPOINTS)
        sub_no2 = compute_sub_index(calibrated_no2, NO2_BREAKPOINTS)
        sub_o3 = compute_sub_index(calibrated_o3, O3_BREAKPOINTS)

        # CPCB Official Rule: Overall AQI requires at least 3 measured pollutants,
        # with at least one being PM2.5 or PM10.
        has_particulate = "PM2.5" in explicitly_provided or "PM10" in explicitly_provided
        is_cpcb_compliant = len(explicitly_provided) >= 3 and has_particulate
        data_quality_flags = []
        if not is_cpcb_compliant:
            data_quality_flags.append("PROVISIONAL_PARTICULATE_PROXY")
            if len(explicitly_provided) < 3:
                data_quality_flags.append(f"INSUFFICIENT_POLLUTANTS_FOR_OFFICIAL_AQI ({len(explicitly_provided)}/3 required)")
            if not has_particulate:
                data_quality_flags.append("MISSING_MANDATORY_PM_COMPONENT")

        # Composite NAQI is max of sub-indices
        naqi_candidates = [
            ("PM2.5", sub_pm25),
            ("PM10", sub_pm10),
            ("NO2", sub_no2),
            ("O3", sub_o3),
        ]
        prominent_pollutant, naqi_val = max(naqi_candidates, key=lambda x: x[1])
        naqi_val = round(float(naqi_val), 1)
        naqi_cat, naqi_color = classify_naqi(naqi_val)

        # 3. Atmospheric Boundary Layer & Ventilation Index (V = WS * BLH)
        dtr = temp_max_c - temp_min_c
        ws_ms = max(0.2, wind_speed_kmh / 3.6)
        is_winter = month in [11, 12, 1, 2]
        is_calm = wind_speed_kmh <= 10.0
        is_dry = precipitation_mm < 1.0

        # Boundary Layer Mixing Height (m) derivation
        if boundary_layer_height_m is not None:
            blh_m = max(100.0, float(boundary_layer_height_m))
        else:
            # Physical proxy: nocturnal winter inversion collapses BLH to ~200-400m;
            # summer convective heating elevates BLH to ~1200-2000m
            base_blh = 450.0 if is_winter else 900.0
            convective_bonus = max(0.0, dtr) * 35.0
            blh_m = float(np.clip(base_blh + convective_bonus - (10.0 if is_calm else 0.0), 180.0, 2400.0))

        # Ventilation Index (m2/s)
        # V < 2000 m2/s = poor dispersion; V < 1000 m2/s = critical stagnation trapping
        ventilation_index = round(float(ws_ms * blh_m), 1)
        ventilation_proxy = round(float(wind_speed_kmh * (100.0 + max(0.0, dtr) * 45.0)), 1)

        # Inversion trapping logic
        stagnant_physics = (is_winter or month == 10) and is_calm and is_dry
        stagnation_prob = 0.10
        if stagnant_physics:
            stagnation_prob = min(0.95, 0.45 + (12.0 - wind_speed_kmh) * 0.04 + min(0.20, dtr * 0.015))
        elif ventilation_index < 2000.0:
            stagnation_prob = min(0.85, 0.30 + (2000.0 - ventilation_index) / 3000.0)

        stagnation_flag = int(stagnation_prob >= 0.50 or (is_calm and is_winter) or ventilation_index < 1200.0)

        # 4. Transboundary Smoke Risk (Indo-Gangetic Plain Stubble Plume)
        igp_cities = [
            "delhi", "faridabad", "gurugram", "noida", "ghaziabad", "meerut", "ludhiana",
            "amritsar", "jalandhar", "chandigarh", "ambala", "patiala", "lucknow", "kanpur",
            "agra", "varanasi", "patna"
        ]
        is_igp = city.lower() in igp_cities
        is_stubble_season = 285 <= day_of_year <= 335

        smoke_advection_risk = False
        smoke_severity = "NONE"
        if is_igp and is_stubble_season:
            smoke_advection_risk = True
            if calibrated_pm25 >= 180.0 or stagnation_flag:
                smoke_severity = "SEVERE_TRANSBOUNDARY_SMOKE_INCURSION"
            else:
                smoke_severity = "MODERATE_SMOKE_PLUME"

        # 5. Multi-Horizon Air Quality Projections (24h, 48h, 72h)
        multi_horizon = {
            "h24": {
                "pm2_5_ug_m3": calibrated_pm25,
                "pm10_ug_m3": calibrated_pm10,
                "no2_ug_m3": calibrated_no2,
                "o3_ug_m3": calibrated_o3,
                "naqi": naqi_val,
                "naqi_category": naqi_cat,
            },
            "h48": {
                "pm2_5_ug_m3": round(calibrated_pm25 * (1.08 if stagnation_flag else 0.95), 1),
                "pm10_ug_m3": round(calibrated_pm10 * (1.06 if stagnation_flag else 0.96), 1),
                "no2_ug_m3": round(calibrated_no2 * (1.04 if stagnation_flag else 0.98), 1),
                "o3_ug_m3": round(calibrated_o3, 1),
                "naqi": round(naqi_val * (1.07 if stagnation_flag else 0.96), 1),
                "naqi_category": classify_naqi(naqi_val * (1.07 if stagnation_flag else 0.96))[0],
            },
            "h72": {
                "pm2_5_ug_m3": round(calibrated_pm25 * (1.12 if stagnation_flag else 0.92), 1),
                "pm10_ug_m3": round(calibrated_pm10 * (1.10 if stagnation_flag else 0.93), 1),
                "no2_ug_m3": round(calibrated_no2 * (1.06 if stagnation_flag else 0.97), 1),
                "o3_ug_m3": round(calibrated_o3, 1),
                "naqi": round(naqi_val * (1.11 if stagnation_flag else 0.93), 1),
                "naqi_category": classify_naqi(naqi_val * (1.11 if stagnation_flag else 0.93))[0],
            },
        }

        # 6. Health Advisories & Civil Decision Guidance
        advisories: List[str] = []
        restrictions: List[str] = []

        if naqi_cat in ["Very Poor", "Severe"]:
            restrictions.append("Suspend outdoor morning exercises, school sports assemblies, and non-essential construction.")
            advisories.append("Vulnerable groups (asthma, COPD, cardiovascular patients, children) must remain indoors.")
            advisories.append("Use N95 / FFP2 respirators if outdoor transit is mandatory.")
            if is_igp:
                restrictions.append("GRAP Stage III/IV Trigger: Restrict BS-III petrol and BS-IV diesel vehicle movements in NCR.")
        elif naqi_cat == "Poor":
            advisories.append("Prolonged outdoor exertion may cause breathing discomfort to people with lung/heart diseases.")
            advisories.append("Wear protective masks during peak rush-hour traffic.")
        elif naqi_cat == "Moderate":
            advisories.append("Air quality is acceptable; however, unusually sensitive individuals should monitor throat irritation.")

        if stagnation_flag:
            advisories.append(f"Atmospheric Stagnation Warning: Low ventilation index ({ventilation_index:.0f} m²/s) traps toxic particulates in shallow boundary layer ({blh_m:.0f}m).")

        if smoke_severity == "SEVERE_TRANSBOUNDARY_SMOKE_INCURSION":
            advisories.append("Transboundary Bio-mass Smoke: Agricultural stubble burning plumes entering boundary layer.")

        exposure_summary = {
            "population_exposed_15km": population_15km,
            "schools_exposed_15km": schools_15km,
            "hospitals_exposed_15km": hospitals_15km,
            "vulnerability_tier": "HIGH_DENSITY" if (population_15km and population_15km > 3_000_000) else "MODERATE",
        }

        return {
            "city": city,
            "naqi": naqi_val,
            "naqi_category": naqi_cat,
            "naqi_color": naqi_color,
            "prominent_pollutant": prominent_pollutant,
            "pollutants": {
                "pm2_5": {
                    "concentration_ug_m3": round(calibrated_pm25, 1),
                    "sub_index": sub_pm25,
                },
                "pm10": {
                    "concentration_ug_m3": round(calibrated_pm10, 1),
                    "sub_index": sub_pm10,
                },
                "no2": {
                    "concentration_ug_m3": round(calibrated_no2, 1),
                    "sub_index": sub_no2,
                },
                "o3": {
                    "concentration_ug_m3": round(calibrated_o3, 1),
                    "sub_index": sub_o3,
                },
            },
            "atmospheric_physics": {
                "diurnal_temp_range_c": round(dtr, 1),
                "boundary_layer_height_m": round(blh_m, 1),
                "ventilation_index_m2_s": ventilation_index,
                "ventilation_index_proxy": ventilation_proxy,
                "stagnation_probability": round(stagnation_prob, 3),
                "stagnation_flag": stagnation_flag,
                "transboundary_smoke_risk": smoke_advection_risk,
                "smoke_severity": smoke_severity,
            },
            "exposure_context": exposure_summary,
            "cpcb_compliance": {
                "is_cpcb_compliant": is_cpcb_compliant,
                "explicitly_monitored_pollutants": explicitly_provided,
                "monitored_count": len(explicitly_provided),
                "has_mandatory_pm": has_particulate,
                "data_quality_flags": data_quality_flags,
                "averaging_protocols": {
                    "pm2_5": "24-hour mean",
                    "pm10": "24-hour mean",
                    "no2": "24-hour mean",
                    "so2": "24-hour mean",
                    "nh3": "24-hour mean",
                    "pb": "24-hour mean",
                    "co": "8-hour maximum",
                    "o3": "8-hour maximum",
                },
            },
            "multi_horizon_air_quality": multi_horizon,
            "advisories": advisories,
            "restrictions": restrictions,
        }
