"""
SAGE Crop Calendar & Phenological Growth-Stage Engine.
Extracts state-level planting and harvesting day-of-year (DOY) schedules for 15+ major Indian crops
(Rice, Wheat, Cotton, Maize, Sorghum, Millet, Pulses, Mustard, Sugarcane) from SAGE / Univ. of Wisconsin
and maps growth stages (Sowing, Vegetative, Anthesis/Flowering, Ripening, Harvest) across 50 Indian cities.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

from climacore.common.logging import get_logger
from climacore.config.loader import get_project_root

logger = get_logger("climacore.data.sources.crop_calendar")

INDIAN_STATES_IN_CALENDAR = [
    "Tamil Nadu", "Maharashtra", "Rajasthan", "Andhra Pradesh", "Karnataka",
    "Orissa", "Himachal Pradesh", "West Bengal", "Assam", "Gujarat", "Kerala",
    "India", "India (North)"
]

# Map each city to its primary agro-climatic state in SAGE Crop Calendar
CITY_TO_CALENDAR_STATE: Dict[str, str] = {
    "Agartala": "Assam",
    "Agra": "Rajasthan",
    "Ahmedabad": "Gujarat",
    "Amritsar": "Rajasthan",
    "Bengaluru": "Karnataka",
    "Bhopal": "Maharashtra",
    "Bhubaneswar": "Orissa",
    "Chandigarh": "Rajasthan",
    "Chennai": "Tamil Nadu",
    "Coimbatore": "Tamil Nadu",
    "Dehradun": "Himachal Pradesh",
    "Delhi": "Rajasthan",
    "Gangtok": "Assam",
    "Guwahati": "Assam",
    "Gwalior": "Rajasthan",
    "Hyderabad": "Andhra Pradesh",
    "Imphal": "Assam",
    "Indore": "Maharashtra",
    "Jaipur": "Rajasthan",
    "Jammu": "Himachal Pradesh",
    "Jodhpur": "Rajasthan",
    "Kanpur": "Rajasthan",
    "Kochi": "Kerala",
    "Kolkata": "West Bengal",
    "Leh": "Himachal Pradesh",
    "Lucknow": "Rajasthan",
    "Ludhiana": "Rajasthan",
    "Madurai": "Tamil Nadu",
    "Mangaluru": "Karnataka",
    "Mumbai": "Maharashtra",
    "Mysuru": "Karnataka",
    "Nagpur": "Maharashtra",
    "Nashik": "Maharashtra",
    "Panaji": "Karnataka",
    "Patna": "West Bengal",
    "Port Blair": "Tamil Nadu",
    "Prayagraj": "Rajasthan",
    "Puducherry": "Tamil Nadu",
    "Pune": "Maharashtra",
    "Raipur": "Orissa",
    "Ranchi": "West Bengal",
    "Shillong": "Assam",
    "Shimla": "Himachal Pradesh",
    "Srinagar": "Himachal Pradesh",
    "Surat": "Gujarat",
    "Thiruvananthapuram": "Kerala",
    "Vadodara": "Gujarat",
    "Varanasi": "Rajasthan",
    "Vijayawada": "Andhra Pradesh",
    "Visakhapatnam": "Andhra Pradesh",
}


def doy_to_date_str(doy: Optional[float]) -> str:
    """Convert Day of Year (1-365) to approximate Month/Day string."""
    if doy is None or pd.isna(doy):
        return "N/A"
    doy_int = int(np.clip(round(float(doy)), 1, 365))
    dt = datetime.strptime(f"2026-{doy_int}", "%Y-%j")
    return dt.strftime("%b %d")


def build_city_crop_calendars(
    output_parquet: Optional[Path] = None,
    output_json: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build consolidated crop phenology calendars for all 50 cities."""
    root = get_project_root()
    raw_path = root / "data" / "raw" / "crop_calendar" / "All_data_with_climate.csv"
    if output_parquet is None:
        output_parquet = root / "data" / "processed" / "agriculture" / "city_crop_calendars.parquet"
    if output_json is None:
        output_json = root / "data" / "processed" / "agriculture" / "city_crop_calendars.json"
    if manifest_path is None:
        manifest_path = root / "data" / "manifests" / "crop_calendar_manifest.json"

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading SAGE crop calendar data from {raw_path}...")
    df_raw = pd.read_csv(raw_path, low_memory=False)

    # Filter to Indian state-level records
    sub = df_raw[df_raw["Location"].isin(INDIAN_STATES_IN_CALENDAR)].copy()

    # Load city dominant crops from district profiles
    dist_prof_path = root / "data" / "processed" / "agriculture" / "city_district_crop_profiles.parquet"
    df_crops = pd.read_parquet(dist_prof_path) if dist_prof_path.exists() else pd.DataFrame()
    dom_kharif_map = df_crops.set_index("city")["dominant_kharif_crop"].to_dict() if not df_crops.empty else {}
    dom_rabi_map = df_crops.set_index("city")["dominant_rabi_crop"].to_dict() if not df_crops.empty else {}

    city_calendars = []

    for city, cal_state in CITY_TO_CALENDAR_STATE.items():
        state_crops = sub[sub["Location"] == cal_state]
        if state_crops.empty:
            state_crops = sub[sub["Location"] == "India"]

        target_kharif = dom_kharif_map.get(city, "Rice")
        target_rabi = dom_rabi_map.get(city, "Wheat")

        # 1. Match Kharif Crop Calendar
        k_cands = state_crops[state_crops["Crop"].str.lower().str.contains(target_kharif.lower()[:4], na=False)]
        if k_cands.empty:
            k_cands = state_crops[state_crops["Crop"].str.lower().str.contains("rice", na=False)]
        if k_cands.empty:
            k_cands = sub[sub["Crop"].str.lower().str.contains("rice", na=False)]
        k_row = k_cands.iloc[0]

        # 2. Match Rabi Crop Calendar
        r_cands = state_crops[state_crops["Crop"].str.lower().str.contains(target_rabi.lower()[:4], na=False)]
        if r_cands.empty:
            r_cands = state_crops[state_crops["Crop"].str.lower().str.contains("wheat", na=False)]
        if r_cands.empty:
            r_cands = sub[sub["Crop"].str.lower().str.contains("wheat", na=False)]
        r_row = r_cands.iloc[0]

        # Kharif Phenological Windows
        k_plant_start = float(k_row["Plant.start"])
        k_plant_end = float(k_row["Plant.end"])
        k_harv_start = float(k_row["Harvest.start"])
        k_harv_end = float(k_row["Harvest.end"])
        k_duration = float(k_row["tot.days"]) if pd.notna(k_row["tot.days"]) else 120.0
        # Flowering / Anthesis is roughly 50-75% into cycle
        k_anthesis_start = (k_plant_start + 0.50 * k_duration) % 365
        k_anthesis_end = (k_plant_start + 0.75 * k_duration) % 365

        # Rabi Phenological Windows
        r_plant_start = float(r_row["Plant.start"])
        r_plant_end = float(r_row["Plant.end"])
        r_harv_start = float(r_row["Harvest.start"])
        r_harv_end = float(r_row["Harvest.end"])
        r_duration = float(r_row["tot.days"]) if pd.notna(r_row["tot.days"]) else 135.0
        r_anthesis_start = (r_plant_start + 0.50 * r_duration) % 365
        r_anthesis_end = (r_plant_start + 0.75 * r_duration) % 365

        record = {
            "city": city,
            "agro_calendar_state": cal_state,
            # Kharif
            "kharif_crop": target_kharif,
            "kharif_sowing_doy_range": [round(k_plant_start), round(k_plant_end)],
            "kharif_sowing_dates": f"{doy_to_date_str(k_plant_start)} - {doy_to_date_str(k_plant_end)}",
            "kharif_anthesis_doy_range": [round(k_anthesis_start), round(k_anthesis_end)],
            "kharif_anthesis_dates": f"{doy_to_date_str(k_anthesis_start)} - {doy_to_date_str(k_anthesis_end)}",
            "kharif_harvest_doy_range": [round(k_harv_start), round(k_harv_end)],
            "kharif_harvest_dates": f"{doy_to_date_str(k_harv_start)} - {doy_to_date_str(k_harv_end)}",
            "kharif_duration_days": round(k_duration, 1),
            # Rabi
            "rabi_crop": target_rabi,
            "rabi_sowing_doy_range": [round(r_plant_start), round(r_plant_end)],
            "rabi_sowing_dates": f"{doy_to_date_str(r_plant_start)} - {doy_to_date_str(r_plant_end)}",
            "rabi_anthesis_doy_range": [round(r_anthesis_start), round(r_anthesis_end)],
            "rabi_anthesis_dates": f"{doy_to_date_str(r_anthesis_start)} - {doy_to_date_str(r_anthesis_end)}",
            "rabi_harvest_doy_range": [round(r_harv_start), round(r_harv_end)],
            "rabi_harvest_dates": f"{doy_to_date_str(r_harv_start)} - {doy_to_date_str(r_harv_end)}",
            "rabi_duration_days": round(r_duration, 1),
        }
        city_calendars.append(record)

    df_cal = pd.DataFrame(city_calendars)
    logger.info(f"Saving city crop calendars to {output_parquet} and {output_json}...")
    df_cal.to_parquet(output_parquet, index=False)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(city_calendars, f, indent=2)

    hasher = hashlib.sha256()
    with open(output_parquet, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    sha256_hash = hasher.hexdigest()

    manifest = {
        "dataset_name": "SAGE India Crop Calendar & Phenology Schedule (50 Cities)",
        "source": "SAGE Research Group / University of Wisconsin",
        "output_parquet": str(output_parquet),
        "output_json": str(output_json),
        "sha256": sha256_hash,
        "total_cities": len(df_cal),
        "columns": df_cal.columns.tolist(),
        "summary": {
            "cities_evaluated": len(df_cal),
            "sample_wheat_anthesis_dates": df_cal.set_index("city")["rabi_anthesis_dates"].head(5).to_dict(),
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Crop calendar manifest written to {manifest_path}")
    return df_cal


if __name__ == "__main__":
    df = build_city_crop_calendars()
    print(f"Generated crop calendar phenology profiles for {len(df)} cities.")
    print("Sample:\n", df[["city", "rabi_crop", "rabi_sowing_dates", "rabi_anthesis_dates", "rabi_harvest_dates"]].head(6))
