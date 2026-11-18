"""
CHETNA Thermal Power Plant Ingestion & Spatial Industrial Airshed Pipeline.
Extracts 760+ thermal power plants (Coal, Gas, Lignite, Nuclear) across India
from IIT Bombay / LSCE CHETNA inventory and computes physical point-source emission footprints,
upstream thermal capacity (MW), annual CO2 loads, and distance-to-closest-plant across 50 Indian cities.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from climacore.common.logging import get_logger
from climacore.config.loader import get_project_root

logger = get_logger("climacore.data.sources.power_plants_chetna")


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized Haversine distance in km."""
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def calculate_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, str]:
    """Calculate compass bearing in degrees (0-360) and 8-point compass sector."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlambda = np.radians(lon2 - lon1)
    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    deg = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0

    sectors = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    sector = sectors[int(round(deg / 45.0))]
    return round(deg, 1), sector


def build_city_power_plant_exposure(
    output_parquet: Optional[Path] = None,
    output_json: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build consolidated power plant point-source exposure metrics for 50 cities."""
    root = get_project_root()
    raw_path = root / "data" / "raw" / "power_plants_chetna" / "CHETNA_Power_Plants_inventory.csv"
    if output_parquet is None:
        output_parquet = root / "data" / "processed" / "power_plants" / "city_power_plant_exposure.parquet"
    if output_json is None:
        output_json = root / "data" / "processed" / "power_plants" / "city_power_plant_exposure.json"
    if manifest_path is None:
        manifest_path = root / "data" / "manifests" / "chetna_power_plants_manifest.json"

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading CHETNA inventory from {raw_path}...")
    df_raw = pd.read_csv(raw_path)

    # Clean and aggregate physical plants
    plants = df_raw.groupby(["plant_name", "latitude", "longitude"]).agg({
        "state_id": "first",
        "fuel_gem": lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else "thermal",
        "technology": lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else "subcritical",
        "installed_capacity_mw": "max",
        "total_co2ton": "mean",
    }).reset_index()

    plants["installed_capacity_mw"] = plants["installed_capacity_mw"].fillna(0.0).astype(float)
    plants["total_co2ton"] = plants["total_co2ton"].fillna(0.0).astype(float)

    # Load 50 cities coordinates
    pop_path = root / "data" / "processed" / "population" / "city_population_exposure.parquet"
    cities_df = pd.read_parquet(pop_path)[["city", "state", "latitude", "longitude"]].drop_duplicates()

    city_records: List[Dict[str, Any]] = []

    p_lats = plants["latitude"].values
    p_lons = plants["longitude"].values

    for _, crow in cities_df.iterrows():
        city = str(crow["city"])
        state = str(crow["state"])
        c_lat = float(crow["latitude"])
        c_lon = float(crow["longitude"])

        dists = haversine_km(c_lat, c_lon, p_lats, p_lons)

        m50 = dists <= 50.0
        m100 = dists <= 100.0
        m250 = dists <= 250.0

        p50 = plants[m50]
        p100 = plants[m100]
        p250 = plants[m250]

        min_idx = int(np.argmin(dists))
        closest = plants.iloc[min_idx]
        closest_bearing, closest_dir = calculate_bearing_deg(c_lat, c_lon, closest["latitude"], closest["longitude"])

        city_records.append({
            "city": city,
            "state": state,
            "plants_within_50km": int(m50.sum()),
            "thermal_capacity_50km_mw": float(p50["installed_capacity_mw"].sum()),
            "annual_co2_50km_million_tons": round(float(p50["total_co2ton"].sum() / 1e6), 3),
            "plants_within_100km": int(m100.sum()),
            "thermal_capacity_100km_mw": float(p100["installed_capacity_mw"].sum()),
            "annual_co2_100km_million_tons": round(float(p100["total_co2ton"].sum() / 1e6), 3),
            "plants_within_250km": int(m250.sum()),
            "thermal_capacity_250km_mw": float(p250["installed_capacity_mw"].sum()),
            "annual_co2_250km_million_tons": round(float(p250["total_co2ton"].sum() / 1e6), 3),
            "closest_plant_name": str(closest["plant_name"]),
            "closest_plant_distance_km": round(float(dists[min_idx]), 1),
            "closest_plant_capacity_mw": float(closest["installed_capacity_mw"]),
            "closest_plant_fuel": str(closest["fuel_gem"]),
            "closest_plant_bearing_deg": closest_bearing,
            "closest_plant_direction": closest_dir,
        })

    df_out = pd.DataFrame(city_records)
    logger.info(f"Saving power plant exposure profiles to {output_parquet} and {output_json}...")
    df_out.to_parquet(output_parquet, index=False)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(city_records, f, indent=2)

    hasher = hashlib.sha256()
    with open(output_parquet, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    sha256_hash = hasher.hexdigest()

    manifest = {
        "dataset_name": "CHETNA Thermal Power Plant Industrial Exposure (50 Cities)",
        "source": "IIT Bombay & LSCE, CEA, Global Energy Monitor",
        "output_parquet": str(output_parquet),
        "output_json": str(output_json),
        "sha256": sha256_hash,
        "total_cities": len(df_out),
        "total_power_plants_cataloged": len(plants),
        "total_generating_capacity_mw": float(plants["installed_capacity_mw"].sum()),
        "summary": {
            "top_airshed_capacities_100km": df_out.sort_values("thermal_capacity_100km_mw", ascending=False)[
                ["city", "thermal_capacity_100km_mw", "plants_within_100km", "annual_co2_100km_million_tons"]
            ].head(8).to_dict(orient="records"),
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"CHETNA power plants manifest written to {manifest_path}")
    return df_out


if __name__ == "__main__":
    df = build_city_power_plant_exposure()
    print(f"Generated industrial power plant exposure for {len(df)} cities.")
    print("Top 8 cities by 100km thermal plant capacity:\n", df[["city", "plants_within_100km", "thermal_capacity_100km_mw", "closest_plant_name", "closest_plant_distance_km", "closest_plant_direction"]].sort_values("thermal_capacity_100km_mw", ascending=False).head(8).to_string())
