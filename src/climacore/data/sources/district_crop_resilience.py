"""
India District Crop & Soil Resilience Ingestion Pipeline.
Extracts 20 soil physicochemical properties (pH, organic carbon, nitrogen, CEC, texture, bulk density)
and dominant multi-season crop profiles from 34,903 historical agricultural district records (1997-2015)
mapped to the agricultural and peri-urban hinterlands of all 50 Indian cities.
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

logger = get_logger("climacore.data.sources.district_crop_resilience")


def load_raw_crop_dataset(raw_path: Optional[Path] = None) -> pd.DataFrame:
    """Load raw India District Crop Dataset."""
    if raw_path is None:
        root = get_project_root()
        raw_path = root / "data" / "raw" / "agriculture" / "india_district_crop_dataset_1997_2015.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"Crop dataset not found at {raw_path}")

    logger.info(f"Loading raw district crop dataset from {raw_path}...")
    return pd.read_csv(raw_path, low_memory=False)


def map_cities_to_agricultural_districts(
    crop_df: pd.DataFrame,
    cities_parquet: Optional[Path] = None,
) -> pd.DataFrame:
    """Map each of the 50 cities to its agricultural district by name alias or nearest centroid."""
    root = get_project_root()
    if cities_parquet is None:
        cities_parquet = root / "data" / "processed" / "population" / "city_population_exposure.parquet"

    cities_df = pd.read_parquet(cities_parquet)[["city", "state", "latitude", "longitude"]].drop_duplicates()
    crop_dists = crop_df[["state", "district", "centroid_lat", "centroid_lon"]].dropna().drop_duplicates()

    CITY_DISTRICT_ALIASES = {
        "Delhi": "Faridabad",
        "Kolkata": "North Twenty Four Pargan*",
        "Ahmedabad": "Ahmadabad",
        "Gangtok": "East District",
        "Chennai": "Thiruvallur",
        "Mumbai": "Thane",
        "Bengaluru": "Bengaluru Rural",
        "Prayagraj": "Allahabad",
        "Mysuru": "Mysore",
        "Varanasi": "Varanasi",
        "Kochi": "Ernakulam",
        "Guwahati": "Kamrup",
        "Bhubaneswar": "Khordha",
        "Agartala": "West Tripura",
        "Shillong": "East Khasi Hills",
        "Mangaluru": "Dakshina Kannada",
        "Panaji": "North Goa",
        "Vijayawada": "Krishna",
        "Visakhapatnam": "Visakhapatnam",
        "Port Blair": "South Andaman",
    }

    mappings = []
    for _, crow in cities_df.iterrows():
        city = str(crow["city"])
        c_lat = float(crow["latitude"])
        c_lon = float(crow["longitude"])

        matched_dist = None
        match_type = "name_match"
        dist_km = 0.0

        # 1. Check explicit alias
        if city in CITY_DISTRICT_ALIASES:
            alias = CITY_DISTRICT_ALIASES[city]
            cand = crop_dists[crop_dists["district"].str.lower() == alias.lower()]
            if len(cand) == 0:
                cand = crop_dists[crop_dists["district"].str.lower().str.contains(alias.lower()[:8], na=False)]
            if len(cand) > 0:
                matched_dist = cand.iloc[0]
                dist_km = float(np.sqrt((c_lat - matched_dist["centroid_lat"])**2 + (c_lon - matched_dist["centroid_lon"])**2) * 111.0)

        # 2. Check exact/substring city name match
        if matched_dist is None:
            named = crop_dists[crop_dists["district"].str.lower().str.contains(city.lower(), na=False)]
            if len(named) > 0:
                matched_dist = named.iloc[0]
                dist_km = float(np.sqrt((c_lat - matched_dist["centroid_lat"])**2 + (c_lon - matched_dist["centroid_lon"])**2) * 111.0)

        # 3. Fallback to nearest geographic centroid
        if matched_dist is None:
            match_type = "geo_nearest"
            dists = np.sqrt((crop_dists["centroid_lat"] - c_lat)**2 + (crop_dists["centroid_lon"] - c_lon)**2) * 111.0
            min_idx = dists.idxmin()
            matched_dist = crop_dists.loc[min_idx]
            dist_km = float(dists[min_idx])

        mappings.append({
            "city": city,
            "agricultural_district": str(matched_dist["district"]),
            "agricultural_state": str(matched_dist["state"]),
            "distance_to_hinterland_km": round(dist_km, 1),
            "match_type": match_type,
        })

    return pd.DataFrame(mappings)


def build_city_crop_resilience_profiles(
    output_parquet: Optional[Path] = None,
    output_json: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build consolidated agricultural soil and crop profiles for all 50 cities."""
    root = get_project_root()
    if output_parquet is None:
        output_parquet = root / "data" / "processed" / "agriculture" / "city_district_crop_profiles.parquet"
    if output_json is None:
        output_json = root / "data" / "processed" / "agriculture" / "city_district_crop_profiles.json"
    if manifest_path is None:
        manifest_path = root / "data" / "manifests" / "crop_dataset_manifest.json"

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    df_crop = load_raw_crop_dataset()
    mappings_df = map_cities_to_agricultural_districts(df_crop)

    city_profiles = []
    for _, mrow in mappings_df.iterrows():
        city = mrow["city"]
        dist = mrow["agricultural_district"]
        dist_records = df_crop[df_crop["district"] == dist]
        if len(dist_records) == 0:
            dist_records = df_crop[df_crop["district"].str.lower().str.contains(dist.lower()[:6], na=False)]

        # Soil properties (depth 0-30 cm)
        soil_ph = float(dist_records["soil_ph_h2o_0_30cm"].median())
        soil_n = float(dist_records["soil_total_n_g_kg_0_30cm"].median())
        soil_soc = float(dist_records["soil_soc_g_kg_0_30cm"].median())
        soil_cec = float(dist_records["soil_cec_cmol_kg_0_30cm"].median())
        soil_clay = float(dist_records["soil_clay_pct_0_30cm"].median())
        soil_sand = float(dist_records["soil_sand_pct_0_30cm"].median())
        soil_silt = float(dist_records["soil_silt_pct_0_30cm"].median())
        soil_bulk_density = float(dist_records["soil_bulk_density_kg_dm3_0_30cm"].median())

        # Soil Water Holding Capacity Index
        whc = round(0.4 * soil_clay + 0.2 * soil_silt + 0.05 * soil_sand + 0.1 * soil_soc, 2)

        # Soil Fertility Index (0-100 score)
        # SOC typical 0-40 g/kg (ideal > 15), Nitrogen 0-4 g/kg (ideal > 1.5), CEC 5-40 (ideal > 20)
        soc_score = min(1.0, soil_soc / 20.0)
        n_score = min(1.0, soil_n / 2.0)
        cec_score = min(1.0, soil_cec / 25.0)
        fertility_index = round(float((soc_score * 0.4 + n_score * 0.35 + cec_score * 0.25) * 100), 1)

        # Cropping patterns
        kharif_crops = dist_records[dist_records["season"] == "Kharif"]["dominant_crop"].value_counts()
        dom_kharif = kharif_crops.index[0] if len(kharif_crops) > 0 else "Rice"

        rabi_crops = dist_records[dist_records["season"] == "Rabi"]["dominant_crop"].value_counts()
        dom_rabi = rabi_crops.index[0] if len(rabi_crops) > 0 else "Wheat"

        all_crops = dist_records["dominant_crop"].value_counts()
        dom_overall = all_crops.index[0] if len(all_crops) > 0 else "Rice"
        top3_crops = ", ".join(all_crops.head(3).index.tolist())

        ann_rain = float(dist_records["annual_rainfall_mm"].median())
        ann_temp = float(dist_records["annual_temp_mean_c"].median())

        city_profiles.append({
            "city": city,
            "agricultural_district": dist,
            "agricultural_state": mrow["agricultural_state"],
            "distance_to_hinterland_km": mrow["distance_to_hinterland_km"],
            "match_type": mrow["match_type"],
            "soil_ph": round(soil_ph, 2),
            "soil_total_nitrogen_g_kg": round(soil_n, 3),
            "soil_organic_carbon_g_kg": round(soil_soc, 2),
            "soil_cec_cmol_kg": round(soil_cec, 2),
            "soil_clay_pct": round(soil_clay, 1),
            "soil_sand_pct": round(soil_sand, 1),
            "soil_silt_pct": round(soil_silt, 1),
            "soil_bulk_density_kg_dm3": round(soil_bulk_density, 3),
            "soil_water_holding_capacity": whc,
            "soil_fertility_index": fertility_index,
            "dominant_kharif_crop": dom_kharif,
            "dominant_rabi_crop": dom_rabi,
            "dominant_crop_overall": dom_overall,
            "top_crops": top3_crops,
            "district_normal_annual_rainfall_mm": round(ann_rain, 1),
            "district_normal_annual_temp_c": round(ann_temp, 1),
        })

    prof_df = pd.DataFrame(city_profiles)
    logger.info(f"Saving crop resilience profiles to {output_parquet} and {output_json}...")
    prof_df.to_parquet(output_parquet, index=False)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(city_profiles, f, indent=2)

    # Manifest
    hasher = hashlib.sha256()
    with open(output_parquet, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    sha256_hash = hasher.hexdigest()

    manifest = {
        "dataset_name": "AgriSense District Soil & Crop Profiles (50 Cities)",
        "source": "India District Crop Dataset 1997-2015, SoilGrids 250m, ICAR / ICRISAT",
        "output_parquet": str(output_parquet),
        "output_json": str(output_json),
        "sha256": sha256_hash,
        "total_cities": len(prof_df),
        "columns": prof_df.columns.tolist(),
        "summary": {
            "top_kharif_crops": prof_df["dominant_kharif_crop"].value_counts().to_dict(),
            "top_rabi_crops": prof_df["dominant_rabi_crop"].value_counts().to_dict(),
            "mean_fertility_index": round(float(prof_df["soil_fertility_index"].mean()), 2),
            "mean_water_holding_capacity": round(float(prof_df["soil_water_holding_capacity"].mean()), 2),
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Crop dataset manifest written to {manifest_path}")
    return prof_df


if __name__ == "__main__":
    df = build_city_crop_resilience_profiles()
    print(f"Extracted crop & soil resilience profiles for {len(df)} cities.")
    print("Sample:\n", df[["city", "agricultural_district", "dominant_kharif_crop", "dominant_rabi_crop", "soil_fertility_index"]].head(8))
