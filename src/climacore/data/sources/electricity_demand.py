"""
Daily Electricity Demand & Grid Thermal Stress Ingestion Pipeline.
Aligns state-level electricity supply, peak load (MW), and unmet load (load shedding)
from POSOCO / CEA telemetry with daily temperature, UTCI thermal stress,
solar radiation, and cooling degree days across 50 Indian cities (2013-2020).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from climacore.common.logging import get_logger
from climacore.config.loader import get_project_root

logger = get_logger("climacore.data.sources.electricity_demand")

CITY_TO_STATE_GRID: Dict[str, str] = {
    "Agartala": "Tripura",
    "Agra": "Uttar Pradesh",
    "Ahmedabad": "Gujarat",
    "Amritsar": "Punjab",
    "Bengaluru": "Karnataka",
    "Bhopal": "Madhya Pradesh",
    "Bhubaneswar": "Odisha",
    "Chandigarh": "Chandigarh",
    "Chennai": "Tamil Nadu",
    "Coimbatore": "Tamil Nadu",
    "Dehradun": "Uttarakhand",
    "Delhi": "Delhi",
    "Gangtok": "Sikkim",
    "Guwahati": "Assam",
    "Gwalior": "Madhya Pradesh",
    "Hyderabad": "Telangana",
    "Imphal": "Manipur",
    "Indore": "Madhya Pradesh",
    "Jaipur": "Rajasthan",
    "Jammu": "Jammu and Kashmir",
    "Jodhpur": "Rajasthan",
    "Kanpur": "Uttar Pradesh",
    "Kochi": "Kerala",
    "Kolkata": "West Bengal",
    "Leh": "Jammu and Kashmir",
    "Lucknow": "Uttar Pradesh",
    "Ludhiana": "Punjab",
    "Madurai": "Tamil Nadu",
    "Mangaluru": "Karnataka",
    "Mumbai": "Maharashtra",
    "Mysuru": "Karnataka",
    "Nagpur": "Maharashtra",
    "Nashik": "Maharashtra",
    "Panaji": "Goa",
    "Patna": "Bihar",
    "Port Blair": "Andaman and Nicobar Islands",
    "Prayagraj": "Uttar Pradesh",
    "Puducherry": "Puducherry",
    "Pune": "Maharashtra",
    "Raipur": "Chhattisgarh",
    "Ranchi": "Jharkhand",
    "Shillong": "Meghalaya",
    "Shimla": "Himachal Pradesh",
    "Srinagar": "Jammu and Kashmir",
    "Surat": "Gujarat",
    "Thiruvananthapuram": "Kerala",
    "Vadodara": "Gujarat",
    "Varanasi": "Uttar Pradesh",
    "Vijayawada": "Andhra Pradesh",
    "Visakhapatnam": "Andhra Pradesh",
}


def _get_weather_file_name(state: str) -> str:
    """Map state name to its corresponding weather CSV filename."""
    if state == "Delhi":
        return "NCT_of_Delhi.csv"
    if state == "Andaman and Nicobar Islands":
        return "Andaman_and_Nicobar.csv"
    return f"{state.replace(' ', '_')}.csv"


def build_aligned_electricity_dataset(
    output_parquet: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build the consolidated 2013-2020 electricity demand & grid stress dataset across 50 cities."""
    root = get_project_root()
    raw_dir = root / "data" / "raw" / "electricity_demand"
    if output_parquet is None:
        output_parquet = root / "data" / "processed" / "electricity" / "city_daily_electricity_stress.parquet"
    if manifest_path is None:
        manifest_path = root / "data" / "manifests" / "electricity_demand_manifest.json"

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load electricity grid telemetry
    energy_csv = raw_dir / "corrected_daily_energy_met_MU.csv"
    peak_csv = raw_dir / "corrected_peak_met_MW.csv"
    unmet_csv = raw_dir / "peak_unmet_MW.csv"

    if not energy_csv.exists() or not peak_csv.exists():
        raise FileNotFoundError(f"Electricity CSV files missing in {raw_dir}")

    logger.info(f"Loading electricity demand files from {raw_dir}...")
    df_energy = pd.read_csv(energy_csv)
    df_peak = pd.read_csv(peak_csv)
    df_unmet = pd.read_csv(unmet_csv)

    df_energy["date"] = pd.to_datetime(df_energy["Date"])
    df_peak["date"] = pd.to_datetime(df_peak["Date"])
    df_unmet["date"] = pd.to_datetime(df_unmet["Date"])

    # Load static city context
    pop_path = root / "data" / "processed" / "population" / "city_population_exposure.parquet"
    df_pop = pd.read_parquet(pop_path) if pop_path.exists() else pd.DataFrame()
    pop_map = df_pop.set_index("city")["population_15km"].to_dict() if not df_pop.empty else {}
    cluster_map = df_pop.set_index("city")["dense_cluster_pop_15km"].to_dict() if not df_pop.empty else {}

    city_records: List[pd.DataFrame] = []

    # Cache state weather dataframes to avoid re-reading for multiple cities in the same state
    state_weather_cache: Dict[str, pd.DataFrame] = {}

    for city, state in CITY_TO_STATE_GRID.items():
        if state not in state_weather_cache:
            w_file = raw_dir / _get_weather_file_name(state)
            if not w_file.exists():
                logger.warning(f"Weather file not found for {state} at {w_file}")
                continue
            w_df = pd.read_csv(w_file)
            w_df["date"] = pd.to_datetime(w_df["date"])
            # Filter to 2013-2020 overlap
            w_df = w_df[(w_df["date"] >= "2013-01-01") & (w_df["date"] <= "2020-12-31")].copy()
            state_weather_cache[state] = w_df

        w_df = state_weather_cache[state].copy()
        if w_df.empty:
            continue

        # Extract state grid columns
        state_energy = df_energy[["date", state]].rename(columns={state: "energy_met_mu"})
        state_peak = df_peak[["date", state]].rename(columns={state: "peak_met_mw"})
        state_unmet = df_unmet[["date", state]].rename(columns={state: "peak_unmet_mw"})

        city_df = w_df.merge(state_energy, on="date", how="inner")
        city_df = city_df.merge(state_peak, on="date", how="inner")
        city_df = city_df.merge(state_unmet, on="date", how="inner")

        city_df["city"] = city
        city_df["state"] = state

        # Temperatures from Kelvin to Celsius
        city_df["temp_min_c"] = (city_df["2m_temperature_min"] - 273.15).astype(np.float32)
        city_df["temp_mean_c"] = (city_df["2m_temperature_mean"] - 273.15).astype(np.float32)
        city_df["temp_max_c"] = (city_df["2m_temperature_max"] - 273.15).astype(np.float32)
        city_df["dewpoint_mean_c"] = (city_df["2m_dewpoint_temperature_mean"] - 273.15).astype(np.float32)
        city_df["utci_mean_c"] = (city_df["utci_mean"] - 273.15).astype(np.float32)
        city_df["utci_max_c"] = (city_df["utci_max"] - 273.15).astype(np.float32)

        # Thermal engineering features
        city_df["cooling_degree_days"] = np.maximum(0.0, city_df["temp_mean_c"] - 18.0).astype(np.float32)
        city_df["heating_degree_days"] = np.maximum(0.0, 18.0 - city_df["temp_mean_c"]).astype(np.float32)
        city_df["is_tropical_night"] = (city_df["temp_min_c"] >= 28.0).astype(int)
        city_df["diurnal_temp_range"] = (city_df["temp_max_c"] - city_df["temp_min_c"]).astype(np.float32)
        city_df["solar_radiation_mj"] = (city_df["surface_solar_radiation_downwards_mean"] / 1e6).astype(np.float32)
        city_df["cloud_cover_pct"] = (city_df["total_cloud_cover_mean"] * 100.0).astype(np.float32)

        # Temporal indicators
        city_df["day_of_week"] = city_df["date"].dt.day_of_week
        city_df["is_weekend"] = city_df["day_of_week"].isin([5, 6]).astype(int)
        city_df["month"] = city_df["date"].dt.month

        # Targets & Multipliers
        median_energy = float(city_df["energy_met_mu"].median()) if not city_df["energy_met_mu"].empty else 1.0
        city_df["grid_stress_ratio"] = (city_df["energy_met_mu"] / max(1.0, median_energy)).astype(np.float32)
        city_df["peak_unmet_mw"] = city_df["peak_unmet_mw"].clip(lower=0.0).fillna(0.0).astype(np.float32)
        city_df["is_grid_deficit_day"] = (city_df["peak_unmet_mw"] >= 50.0).astype(int)

        # Context
        city_df["population_15km"] = float(pop_map.get(city, 2_000_000))
        city_df["dense_cluster_pop_15km"] = float(cluster_map.get(city, 500_000))

        # Keep clean subset of features
        keep_cols = [
            "city", "state", "date", "temp_min_c", "temp_mean_c", "temp_max_c",
            "dewpoint_mean_c", "utci_mean_c", "utci_max_c", "cooling_degree_days",
            "heating_degree_days", "is_tropical_night", "diurnal_temp_range",
            "solar_radiation_mj", "cloud_cover_pct", "day_of_week", "is_weekend", "month",
            "population_15km", "dense_cluster_pop_15km",
            "energy_met_mu", "peak_met_mw", "peak_unmet_mw",
            "grid_stress_ratio", "is_grid_deficit_day",
        ]
        city_records.append(city_df[keep_cols])

    df_consolidated = pd.concat(city_records, ignore_index=True)
    # Drop records with missing energy telemetry (e.g. Andaman microgrid or pre-2014 Telangana)
    df_consolidated = df_consolidated.dropna(subset=["energy_met_mu", "peak_met_mw", "grid_stress_ratio"]).copy()
    df_consolidated = df_consolidated.sort_values(["city", "date"]).reset_index(drop=True)

    logger.info(f"Saving aligned electricity dataset ({len(df_consolidated):,} rows) to {output_parquet}...")
    df_consolidated.to_parquet(output_parquet, index=False)

    hasher = hashlib.sha256()
    with open(output_parquet, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    sha256_hash = hasher.hexdigest()

    manifest = {
        "dataset_name": "India Daily Electricity Demand & Grid Heat Stress (2013-2020)",
        "source": "University of Reading / Newcastle University (Zenodo), POSOCO, CEA, ERA5-Land",
        "output_file": str(output_parquet),
        "sha256": sha256_hash,
        "total_records": len(df_consolidated),
        "total_cities": int(df_consolidated["city"].nunique()),
        "date_range": [str(df_consolidated["date"].min()), str(df_consolidated["date"].max())],
        "mean_energy_met_mu": round(float(df_consolidated["energy_met_mu"].mean()), 2),
        "mean_peak_met_mw": round(float(df_consolidated["peak_met_mw"].mean()), 2),
        "total_deficit_days": int(df_consolidated["is_grid_deficit_day"].sum()),
        "features": [
            "temp_min_c", "temp_mean_c", "temp_max_c", "dewpoint_mean_c",
            "utci_mean_c", "utci_max_c", "cooling_degree_days", "heating_degree_days",
            "is_tropical_night", "diurnal_temp_range", "solar_radiation_mj",
            "cloud_cover_pct", "day_of_week", "is_weekend", "month",
            "population_15km", "dense_cluster_pop_15km",
        ],
        "targets": ["grid_stress_ratio", "peak_met_mw", "is_grid_deficit_day"],
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Electricity demand manifest written to {manifest_path}")
    return df_consolidated


if __name__ == "__main__":
    df = build_aligned_electricity_dataset()
    print(f"Built electricity stress dataset with {len(df):,} records across {df['city'].nunique()} cities.")
    print("Sample:\n", df[["city", "date", "temp_max_c", "cooling_degree_days", "energy_met_mu", "grid_stress_ratio", "is_grid_deficit_day"]].head(6))
