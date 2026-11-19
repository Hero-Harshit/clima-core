"""
HydroRIVERS Asian River Network Feature Extraction & Spatial Hydrology Engine.
Extracts riverine reach geometry, stream orders, upstream drainage catchments,
and long-term average discharge for Indian urban centers from HydroSHEDS HydroRIVERS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import shapefile

from climacore.common.logging import get_logger
from climacore.config.loader import get_project_root

logger = get_logger("climacore.features.hydrorivers")

# India bounding box: lon in [68.0, 98.0], lat in [6.0, 38.0]
INDIA_BBOX = (68.0, 6.0, 98.0, 38.0)


def extract_city_river_profiles(
    cities_csv_path: Optional[Path] = None,
    shapefile_path: Optional[Path] = None,
    output_json_path: Optional[Path] = None,
    search_radius_deg: float = 0.5,  # ~55 km search radius
) -> Dict[str, Dict[str, Any]]:
    """
    Extract spatial river reach metrics for all canonical 50 Indian cities in a single pass.
    Saves to: data/processed/city_hydrorivers_profiles.json
    """
    root = get_project_root()
    if cities_csv_path is None:
        cities_csv_path = root / "data" / "raw" / "india_weather_air_quality" / "cities.csv"
    if shapefile_path is None:
        shapefile_path = root / "data" / "raw" / "hydrorivers_asia" / "HydroRIVERS_v10_as_shp" / "HydroRIVERS_v10_as.shp"
    if output_json_path is None:
        output_json_path = root / "data" / "processed" / "city_hydrorivers_profiles.json"

    if not shapefile_path.exists():
        raise FileNotFoundError(f"HydroRIVERS shapefile not found at {shapefile_path}")

    cities_df = pd.read_csv(cities_csv_path)
    logger.info(f"Extracting river network profiles for {len(cities_df)} cities from HydroRIVERS...")

    # Prepare city targets
    cities = []
    for _, row in cities_df.iterrows():
        c_name = str(row["city"]).strip()
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        cities.append({
            "city": c_name,
            "latitude": lat,
            "longitude": lon,
            "min_lon": lon - search_radius_deg,
            "max_lon": lon + search_radius_deg,
            "min_lat": lat - search_radius_deg,
            "max_lat": lat + search_radius_deg,
            "min_dist_km": float("inf"),
            "nearest_record": None,
            "major_min_dist_km": float("inf"),
            "nearest_major_record": None,
        })

    sf = shapefile.Reader(str(shapefile_path))

    # Single sequential pass over shapes
    for shape_rec in sf.iterShapeRecords():
        s_bbox = shape_rec.shape.bbox
        # Quick skip outside India
        if (
            s_bbox[0] > INDIA_BBOX[2] or
            s_bbox[2] < INDIA_BBOX[0] or
            s_bbox[1] > INDIA_BBOX[3] or
            s_bbox[3] < INDIA_BBOX[1]
        ):
            continue

        # Check candidate cities
        for c in cities:
            if (
                s_bbox[0] <= c["max_lon"] and
                s_bbox[2] >= c["min_lon"] and
                s_bbox[1] <= c["max_lat"] and
                s_bbox[3] >= c["min_lat"]
            ):
                pts = np.asarray(shape_rec.shape.points)
                if len(pts) == 0:
                    continue
                d_lat = (pts[:, 1] - c["latitude"]) * 111.0
                d_lon = (pts[:, 0] - c["longitude"]) * 111.0 * np.cos(np.radians(c["latitude"]))
                dist = float(np.min(np.sqrt(d_lat**2 + d_lon**2)))

                rec_dict = shape_rec.record.as_dict()

                # Update overall nearest river
                if dist < c["min_dist_km"]:
                    c["min_dist_km"] = dist
                    c["nearest_record"] = rec_dict

                # Check if it qualifies as major river (Strahler order >= 3 or discharge >= 5 m3/s)
                strahler = rec_dict.get("ORD_STRA", 1)
                discharge = rec_dict.get("DIS_AV_CMS", 0.0)
                if strahler >= 3 or discharge >= 5.0:
                    if dist < c["major_min_dist_km"]:
                        c["major_min_dist_km"] = dist
                        c["nearest_major_record"] = rec_dict

    profiles: Dict[str, Dict[str, Any]] = {}
    for c in cities:
        c_name = c["city"]
        near = c["nearest_record"] or {}
        major = c["nearest_major_record"] or near

        profiles[c_name.lower()] = {
            "city": c_name,
            "latitude": c["latitude"],
            "longitude": c["longitude"],
            "nearest_river": {
                "distance_km": round(c["min_dist_km"], 2) if c["min_dist_km"] < float("inf") else 50.0,
                "river_id": near.get("HYRIV_ID"),
                "strahler_order": near.get("ORD_STRA", 1),
                "average_discharge_m3s": round(float(near.get("DIS_AV_CMS", 0.0)), 3),
                "upstream_catchment_sqkm": round(float(near.get("CATCH_SKM", 0.0)), 1),
                "reach_length_km": round(float(near.get("LENGTH_KM", 0.0)), 2),
                "distance_to_ocean_km": round(float(near.get("DIST_DN_KM", 0.0)), 1),
            },
            "nearest_major_river": {
                "distance_km": round(c["major_min_dist_km"], 2) if c["major_min_dist_km"] < float("inf") else 50.0,
                "river_id": major.get("HYRIV_ID"),
                "strahler_order": major.get("ORD_STRA", 1),
                "average_discharge_m3s": round(float(major.get("DIS_AV_CMS", 0.0)), 3),
                "upstream_catchment_sqkm": round(float(major.get("CATCH_SKM", 0.0)), 1),
                "reach_length_km": round(float(major.get("LENGTH_KM", 0.0)), 2),
                "distance_to_ocean_km": round(float(major.get("DIST_DN_KM", 0.0)), 1),
            },
            "fluvial_flood_susceptibility_tier": (
                "VERY_HIGH" if c["major_min_dist_km"] <= 3.0 and float(major.get("DIS_AV_CMS", 0.0)) >= 50.0
                else "HIGH" if c["major_min_dist_km"] <= 7.0 and float(major.get("DIS_AV_CMS", 0.0)) >= 10.0
                else "MODERATE" if c["min_dist_km"] <= 10.0
                else "LOW"
            ),
        }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)

    logger.info(f"Successfully extracted and saved river profiles for {len(profiles)} cities to {output_json_path}")
    return profiles


_CITY_RIVER_PROFILES_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def get_city_river_profile(city: str) -> Dict[str, Any]:
    """Retrieve pre-computed river hydrology profile for a city."""
    global _CITY_RIVER_PROFILES_CACHE
    if _CITY_RIVER_PROFILES_CACHE is None:
        root = get_project_root()
        prof_file = root / "data" / "processed" / "city_hydrorivers_profiles.json"
        if not prof_file.exists():
            _CITY_RIVER_PROFILES_CACHE = extract_city_river_profiles(output_json_path=prof_file)
        else:
            with open(prof_file, "r", encoding="utf-8") as f:
                _CITY_RIVER_PROFILES_CACHE = json.load(f)

    city_key = city.strip().lower()
    if city_key in _CITY_RIVER_PROFILES_CACHE:
        return _CITY_RIVER_PROFILES_CACHE[city_key]

    # Fallback default
    return {
        "city": city,
        "nearest_river": {
            "distance_km": 15.0,
            "strahler_order": 2,
            "average_discharge_m3s": 5.0,
            "upstream_catchment_sqkm": 250.0,
            "reach_length_km": 5.0,
            "distance_to_ocean_km": 500.0,
        },
        "nearest_major_river": {
            "distance_km": 25.0,
            "strahler_order": 4,
            "average_discharge_m3s": 25.0,
            "upstream_catchment_sqkm": 2500.0,
            "reach_length_km": 12.0,
            "distance_to_ocean_km": 500.0,
        },
        "fluvial_flood_susceptibility_tier": "MODERATE",
    }
