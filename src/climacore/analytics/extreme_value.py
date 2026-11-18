"""
Extreme Value Theory (EVT) Statistics Engine.
Replaces naive daily observation percentile mappings with Generalized Extreme Value (GEV)
distributions fitted to 74 years of IMD Temperature (1951-2024) and 46 years of IMD Rainfall (1980-2025)
Annual Maximum Series (AMS) across 50 Indian urban centers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from climacore.common.logging import get_logger
from climacore.config.loader import get_project_root

logger = get_logger("climacore.analytics.extreme_value")

_GEV_PROFILES_CACHE: Optional[Dict[str, Dict[str, Any]]] = None

RETURN_PERIODS: List[int] = [2, 5, 10, 25, 50, 100]


def fit_gev_ams(
    values: np.ndarray,
    n_bootstrap: int = 100,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Fit Generalized Extreme Value distribution to an Annual Maximum Series (AMS)
    and compute return levels with 95% bootstrap confidence intervals.
    """
    clean_vals = values[~np.isnan(values)]
    if len(clean_vals) < 10:
        raise ValueError(f"Insufficient AMS samples for GEV fit: {len(clean_vals)}")

    # Fit GEV: SciPy genextreme has shape c = -xi
    c, loc, scale = stats.genextreme.fit(clean_vals)

    return_levels: Dict[str, float] = {}
    ci_lower: Dict[str, float] = {}
    ci_upper: Dict[str, float] = {}

    for T in RETURN_PERIODS:
        p = 1.0 - (1.0 / float(T))
        rl = float(stats.genextreme.ppf(p, c, loc=loc, scale=scale))
        return_levels[f"T_{T}yr"] = round(rl, 2)

    # Parametric/non-parametric bootstrap for 95% Confidence Intervals
    rng = np.random.default_rng(seed)
    boot_levels: Dict[int, List[float]] = {T: [] for T in RETURN_PERIODS}

    n_samples = len(clean_vals)
    for _ in range(n_bootstrap):
        sample = rng.choice(clean_vals, size=n_samples, replace=True)
        try:
            b_c, b_loc, b_scale = stats.genextreme.fit(sample)
            for T in RETURN_PERIODS:
                p = 1.0 - (1.0 / float(T))
                boot_levels[T].append(float(stats.genextreme.ppf(p, b_c, loc=b_loc, scale=b_scale)))
        except Exception:
            continue

    for T in RETURN_PERIODS:
        arr = np.array(boot_levels[T])
        if len(arr) >= 20:
            ci_lower[f"T_{T}yr"] = round(float(np.percentile(arr, 2.5)), 2)
            ci_upper[f"T_{T}yr"] = round(float(np.percentile(arr, 97.5)), 2)
        else:
            ci_lower[f"T_{T}yr"] = return_levels[f"T_{T}yr"]
            ci_upper[f"T_{T}yr"] = return_levels[f"T_{T}yr"]

    return {
        "sample_count": len(clean_vals),
        "ams_mean": round(float(np.mean(clean_vals)), 2),
        "ams_max": round(float(np.max(clean_vals)), 2),
        "ams_min": round(float(np.min(clean_vals)), 2),
        "params": {
            "c_shape": round(float(c), 4),
            "xi_evt_shape": round(float(-c), 4),
            "loc": round(float(loc), 3),
            "scale": round(float(scale), 3),
            "distribution_type": "Weibull (Upper Bounded)" if c > 0.05 else ("Frechet (Heavy Tailed)" if c < -0.05 else "Gumbel"),
        },
        "return_levels": return_levels,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
    }


def build_city_gev_profiles(
    output_json: Optional[Path] = None,
    output_parquet: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Extract AMS and fit GEV models for all 50 cities across 74 years of IMD temperature
    and 46 years of IMD rainfall.
    """
    root = get_project_root()
    if output_json is None:
        output_json = root / "data" / "processed" / "analytics" / "city_gev_return_periods.json"
    if output_parquet is None:
        output_parquet = root / "data" / "processed" / "analytics" / "city_gev_return_periods.parquet"
    if manifest_path is None:
        manifest_path = root / "data" / "manifests" / "gev_return_periods_manifest.json"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = root / "data" / "processed" / "imd_temperature_74y_50_cities.parquet"
    rain_path = root / "data" / "processed" / "imd_rainfall_50_cities.parquet"

    logger.info(f"Loading 74y IMD temperature data from {temp_path}...")
    df_temp = pd.read_parquet(temp_path)
    logger.info(f"Loading 46y IMD rainfall data from {rain_path}...")
    df_rain = pd.read_parquet(rain_path)
    df_rain["year"] = pd.to_datetime(df_rain["date"]).dt.year

    cities = sorted(df_temp["city"].unique())
    logger.info(f"Computing GEV Annual Maximum Series for {len(cities)} cities...")

    profiles: Dict[str, Dict[str, Any]] = {}
    flat_rows: List[Dict[str, Any]] = []

    for city in cities:
        # 1. Temperature AMS (74 years, 1951-2024)
        sub_t = df_temp[df_temp["city"] == city]
        t_ams = sub_t.groupby("year")["imd_tmax_c"].max().dropna().values
        temp_gev = fit_gev_ams(t_ams, n_bootstrap=100)

        # 2. Rainfall AMS (46 years, 1980-2025)
        sub_r = df_rain[df_rain["city"] == city]
        r_ams = sub_r.groupby("year")["imd_rainfall_mm"].max().dropna().values
        if len(r_ams) < 10:
            if city.lower() == "chennai":
                logger.info(f"Imputing coastal rainfall AMS for {city} from Coromandel coastal proxy (Puducherry)...")
                sub_proxy = df_rain[df_rain["city"].str.lower() == "puducherry"]
                r_ams = sub_proxy.groupby("year")["imd_rainfall_mm"].max().dropna().values
            else:
                logger.info(f"Imputing island rainfall AMS for {city} from coastal maritime proxy (Kochi)...")
                sub_proxy = df_rain[df_rain["city"].str.lower() == "kochi"]
                r_ams = sub_proxy.groupby("year")["imd_rainfall_mm"].max().dropna().values

        rain_gev = fit_gev_ams(r_ams, n_bootstrap=100)

        profile = {
            "city": city,
            "temperature_gev": temp_gev,
            "rainfall_gev": rain_gev,
        }
        profiles[city.lower()] = profile

        flat_row = {
            "city": city,
            "temp_ams_mean_c": temp_gev["ams_mean"],
            "temp_ams_max_c": temp_gev["ams_max"],
            "temp_c_shape": temp_gev["params"]["c_shape"],
            "temp_T_5yr_c": temp_gev["return_levels"]["T_5yr"],
            "temp_T_10yr_c": temp_gev["return_levels"]["T_10yr"],
            "temp_T_25yr_c": temp_gev["return_levels"]["T_25yr"],
            "temp_T_50yr_c": temp_gev["return_levels"]["T_50yr"],
            "rain_ams_mean_mm": rain_gev["ams_mean"],
            "rain_ams_max_mm": rain_gev["ams_max"],
            "rain_c_shape": rain_gev["params"]["c_shape"],
            "rain_T_5yr_mm": rain_gev["return_levels"]["T_5yr"],
            "rain_T_10yr_mm": rain_gev["return_levels"]["T_10yr"],
            "rain_T_25yr_mm": rain_gev["return_levels"]["T_25yr"],
            "rain_T_50yr_mm": rain_gev["return_levels"]["T_50yr"],
        }
        flat_rows.append(flat_row)

    # Save JSON and Parquet
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(list(profiles.values()), f, indent=2)

    df_out = pd.DataFrame(flat_rows)
    df_out.to_parquet(output_parquet, index=False)
    logger.info(f"Saved GEV return periods to {output_json} and {output_parquet}")

    # Generate Manifest
    h = hashlib.sha256()
    with open(output_parquet, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)

    manifest = {
        "dataset_name": "Clima-Core Extreme Value Theory (EVT) Return Periods",
        "methodology": "Generalized Extreme Value (GEV) fitted to Annual Maximum Series (AMS)",
        "temperature_source": "IMD Pune 1.0° Daily Temperature Grids (1951-2024, 74 years)",
        "rainfall_source": "IMD Pune 0.25° Daily Rainfall Grids (1980-2025, 46 years)",
        "total_cities": len(cities),
        "return_periods": RETURN_PERIODS,
        "output_file": str(output_parquet),
        "sha256": h.hexdigest(),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return profiles


def load_gev_profiles() -> Dict[str, Dict[str, Any]]:
    """Load cached GEV return period profiles."""
    global _GEV_PROFILES_CACHE
    if _GEV_PROFILES_CACHE is not None:
        return _GEV_PROFILES_CACHE

    root = get_project_root()
    json_path = root / "data" / "processed" / "analytics" / "city_gev_return_periods.json"
    if not json_path.exists():
        _GEV_PROFILES_CACHE = build_city_gev_profiles(output_json=json_path)
        return _GEV_PROFILES_CACHE

    with open(json_path, "r", encoding="utf-8") as f:
        prof_list = json.load(f)
        _GEV_PROFILES_CACHE = {p["city"].lower(): p for p in prof_list}

    return _GEV_PROFILES_CACHE


def evaluate_return_period(
    city: str,
    value: float,
    hazard_type: str = "temperature"
) -> Dict[str, Any]:
    """
    Calculate the precise GEV return period and statistical rarity category
    for an observed or forecasted temperature or rainfall event in a specific city.
    """
    profiles = load_gev_profiles()
    city_key = city.strip().lower()
    prof = profiles.get(city_key)

    if prof is None:
        return {
            "city": city,
            "hazard_type": hazard_type,
            "value": value,
            "estimated_return_period_years": None,
            "rarity_category": "TYPICAL_SEASONAL_VARIABILITY",
            "gev_exceedance_probability": None,
            "t_10yr_level": None,
            "t_25yr_level": None,
        }

    sub_gev = prof["temperature_gev"] if hazard_type.lower() == "temperature" else prof["rainfall_gev"]
    params = sub_gev["params"]
    c, loc, scale = params["c_shape"], params["loc"], params["scale"]
    ret_levels = sub_gev["return_levels"]

    # CDF gives non-exceedance probability p
    val = float(value)
    p = float(stats.genextreme.cdf(val, c, loc=loc, scale=scale))
    p = float(np.clip(p, 0.0001, 0.9999))

    exceedance_prob = 1.0 - p
    if exceedance_prob <= 0.001:
        t_years = 100.0
    elif exceedance_prob >= 0.50:
        t_years = 1.0
    else:
        t_years = round(float(1.0 / exceedance_prob), 1)

    # Statistical Rarity Tier
    if t_years >= 50.0:
        rarity_desc = "UNPRECEDENTED_HISTORIC_EXTREME (>= 1-in-50 Year Event)"
        risk_level = "CRITICAL_RECORD"
    elif t_years >= 25.0:
        rarity_desc = "VERY_RARE_CLIMATE_EXTREME (>= 1-in-25 Year Event)"
        risk_level = "VERY_RARE"
    elif t_years >= 10.0:
        rarity_desc = "SEVERE_CLIMATE_ANOMALY (>= 1-in-10 Year Event)"
        risk_level = "SEVERE"
    elif t_years >= 5.0:
        rarity_desc = "SIGNIFICANT_ANOMALY (1-in-5 to 1-in-10 Year Event)"
        risk_level = "ELEVATED"
    elif t_years >= 2.0:
        rarity_desc = "TOP_DECIMAL_ANOMALY (1-in-2 to 1-in-5 Year Event)"
        risk_level = "MODERATE"
    else:
        rarity_desc = "TYPICAL_SEASONAL_VARIABILITY (< 1-in-2 Year Event)"
        risk_level = "NORMAL"

    # Extrapolation uncertainty check: return period exceeds observational history
    # 74 years for IMD temperature, 46 years for IMD rainfall
    obs_years = 74 if hazard_type.lower() == "temperature" else 46
    is_extrapolated = bool(t_years > obs_years)

    return {
        "city": prof["city"],
        "hazard_type": hazard_type,
        "value": val,
        "estimated_return_period_years": t_years,
        "rarity_category": rarity_desc,
        "risk_level": risk_level,
        "c_scipy": round(float(c), 4),
        "xi_evt": round(float(-c), 4),
        "tail_behavior": params.get("distribution_type", "GEV"),
        "extrapolation_uncertainty_flag": is_extrapolated,
        "observational_history_years": obs_years,
        "gev_exceedance_probability": round(exceedance_prob, 4),
        "t_5yr_level": ret_levels["T_5yr"],
        "t_10yr_level": ret_levels["T_10yr"],
        "t_25yr_level": ret_levels["T_25yr"],
        "t_50yr_level": ret_levels["T_50yr"],
        "ci_95_t10yr": [sub_gev["ci_95_lower"].get("T_10yr"), sub_gev["ci_95_upper"].get("T_10yr")],
        "ci_95_t25yr": [sub_gev["ci_95_lower"].get("T_25yr"), sub_gev["ci_95_upper"].get("T_25yr")],
        "ci_95_t50yr": [sub_gev["ci_95_lower"].get("T_50yr"), sub_gev["ci_95_upper"].get("T_50yr")],
    }


if __name__ == "__main__":
    profiles = build_city_gev_profiles()
    print("GEV Profiles Built Successfully for 50 Cities!")
    for test_city in ["Delhi", "Nagpur", "Mumbai", "Bengaluru"]:
        res_t = evaluate_return_period(test_city, 46.5, hazard_type="temperature")
        print(f"{test_city} 46.5°C -> {res_t['rarity_category']} (T = {res_t['estimated_return_period_years']} yrs)")
        res_r = evaluate_return_period(test_city, 120.0, hazard_type="rainfall")
        print(f"{test_city} 120mm Rain -> {res_r['rarity_category']} (T = {res_r['estimated_return_period_years']} yrs)")
