"""NDMA & State Heat Action Plan (HAP) City Alert Triggers Registry.

Codifies the official National Disaster Management Authority (NDMA) Heatwave Guidelines
and city-specific Heat Action Plans (Ahmedabad AMC, Odisha OSDMA, Delhi DDMA, etc.)
across all 50 Indian cities categorized by bio-climatic terrain regimes (Plains, Coastal, Hills).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from climacore.common.logging import get_logger
from climacore.common.provenance import compute_sha256
from climacore.config.loader import get_project_root

logger = get_logger("climacore.data.sources.ndma_hap")

COASTAL_CITIES = {
    "Mumbai", "Kolkata", "Chennai", "Surat", "Vadodara",
    "Visakhapatnam", "Kochi", "Thiruvananthapuram", "Bhubaneswar",
    "Mangaluru", "Puducherry", "Panaji", "Port Blair"
}

HILL_CITIES = {
    "Srinagar", "Dehradun", "Shimla", "Jammu",
    "Shillong", "Imphal", "Gangtok", "Leh"
}

# Special Pioneer Municipal Heat Action Plans
SPECIAL_HAP_CITIES = {
    "Ahmedabad": {
        "hap_framework": "Ahmedabad Municipal Corporation (AMC) Pioneer HAP",
        "yellow_alert_range_c": [41.1, 43.0],
        "orange_alert_range_c": [43.1, 44.9],
        "red_alert_threshold_c": 45.0,
        "special_mandates": [
            "White reflective 'Cool Roof' deployment in dense informal settlements",
            "Mandatory night-time bus stand water misting",
            "108 GVK-EMRI ambulance pre-positioning at high-density labor intersections"
        ]
    },
    "Bhubaneswar": {
        "hap_framework": "Odisha State Disaster Management Authority (OSDMA) HAP",
        "yellow_alert_range_c": [37.0, 39.9],
        "orange_alert_range_c": [40.0, 42.9],
        "red_alert_threshold_c": 43.0,
        "humidity_adjusted_trigger": True,
        "special_mandates": [
            "Public transport services suspended between 11:30 AM and 3:30 PM under Red Alert",
            "Jal Chhatras (free drinking water kiosks) at every public bus stand and market",
            "Mandatory paid rest breaks for MGNREGS and construction workers"
        ]
    },
    "Delhi": {
        "hap_framework": "Delhi Disaster Management Authority (DDMA) Graded HAP",
        "yellow_alert_range_c": [40.0, 42.9],
        "orange_alert_range_c": [43.0, 44.9],
        "red_alert_threshold_c": 45.0,
        "special_mandates": [
            "Power DISCOMs prohibited from unscheduled maintenance outages during peak CDD hours",
            "Anti-smog guns deployed on ring roads for evaporative microclimate cooling",
            "Hospital emergency triage heatstroke protocols active across all GNCTD facilities"
        ]
    }
}


def build_ndma_hap_triggers() -> Dict[str, Any]:
    """Build the comprehensive NDMA and State Heat Action Plan triggers for 50 cities."""
    root = get_project_root()
    cities_file = root / "data" / "raw" / "india_weather_air_quality" / "cities.csv"
    cities_df = pd.read_csv(cities_file)

    city_records: List[Dict[str, Any]] = []

    for _, row in cities_df.iterrows():
        c_name = row["city"]
        state = row["state"]
        lat = float(row["latitude"])
        lon = float(row["longitude"])

        if c_name in SPECIAL_HAP_CITIES:
            spec = SPECIAL_HAP_CITIES[c_name]
            terrain = "Special Municipal HAP"
            yellow = spec["yellow_alert_range_c"]
            orange = spec["orange_alert_range_c"]
            red = spec["red_alert_threshold_c"]
            framework = spec["hap_framework"]
            mandates = spec["special_mandates"]
        elif c_name in HILL_CITIES:
            terrain = "Hills / Mountainous Region"
            yellow = [30.0, 32.9]
            orange = [33.0, 34.9]
            red = 35.0
            framework = "NDMA National Heatwave Guidelines (Hills Regime)"
            mandates = [
                "Early advisory for tourists and high-altitude trekkers",
                "Forest fire rapid response teams on high alert",
                "Horticultural apple / fruit orchard moisture conservation advisories"
            ]
        elif c_name in COASTAL_CITIES:
            terrain = "Coastal Sub-Tropical"
            yellow = [37.0, 38.9]
            orange = [39.0, 41.9]
            red = 42.0
            framework = "NDMA National Heatwave Guidelines (Coastal Regime)"
            mandates = [
                "Fishermen marine weather advisory & shoreline high-humidity heat index monitoring",
                "ORS distribution at major ferry terminals and fish markets",
                "Harbor / dockyard cargo handler work-rest cycling"
            ]
        else:
            terrain = "Plains / Continental Interior"
            yellow = [40.0, 42.9]
            orange = [43.0, 44.9]
            red = 45.0
            framework = "NDMA National Heatwave Guidelines (Plains Regime)"
            mandates = [
                "Construction activities halted between 12:00 PM and 4:00 PM",
                "School morning shifts concluding by 11:30 AM",
                "Dedicated municipal water tankers dispatched to dense informal urban settlements"
            ]

        city_records.append({
            "city": c_name,
            "state": state,
            "latitude": lat,
            "longitude": lon,
            "terrain_regime": terrain,
            "hap_framework": framework,
            "threshold_yellow_alert_c": yellow,
            "threshold_orange_alert_c": orange,
            "threshold_red_alert_c": red,
            "departure_heatwave_c": 4.5,
            "departure_severe_heatwave_c": 6.4,
            "mandatory_action_protocols": mandates
        })

    proc_dir = root / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    json_path = proc_dir / "ndma_hap_triggers.json"
    parquet_path = proc_dir / "ndma_hap_triggers.parquet"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(city_records, f, indent=2)

    df_out = pd.DataFrame(city_records)
    df_out["threshold_yellow_min_c"] = df_out["threshold_yellow_alert_c"].apply(lambda x: x[0])
    df_out["threshold_yellow_max_c"] = df_out["threshold_yellow_alert_c"].apply(lambda x: x[1])
    df_out["threshold_orange_min_c"] = df_out["threshold_orange_alert_c"].apply(lambda x: x[0])
    df_out["threshold_orange_max_c"] = df_out["threshold_orange_alert_c"].apply(lambda x: x[1])
    df_out["mandatory_actions_str"] = df_out["mandatory_action_protocols"].apply(lambda l: "; ".join(l))
    
    clean_cols = [
        "city", "state", "latitude", "longitude", "terrain_regime", "hap_framework",
        "threshold_yellow_min_c", "threshold_yellow_max_c",
        "threshold_orange_min_c", "threshold_orange_max_c",
        "threshold_red_alert_c", "departure_heatwave_c", "departure_severe_heatwave_c",
        "mandatory_actions_str"
    ]
    df_out[clean_cols].to_parquet(parquet_path, index=False)

    manifest_path = root / "data" / "manifests" / "ndma_hap_manifest.json"
    manifest = {
        "source_id": "ndma_heat_action_plan_triggers",
        "display_name": "NDMA & State Heat Action Plan (HAP) City Alert Triggers",
        "provider": "National Disaster Management Authority (NDMA) & Municipal Corporations",
        "spatial_coverage": "50 Major Indian Urban Centers across 28 States and Union Territories",
        "cities_covered": len(city_records),
        "json_path": str(json_path.name),
        "parquet_path": str(parquet_path.name),
        "sha256_json": compute_sha256(json_path),
        "sha256_parquet": compute_sha256(parquet_path),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "color_code_alert_standards": {
            "Yellow": "Watch and Update: Moderate heat risk. Public awareness and hydration advisories.",
            "Orange": "Be Prepared: High heat risk for vulnerable groups. School shifts adjusted, outdoor labor restricted.",
            "Red": "Take Action: Extreme heat emergency. Non-essential outdoor labor prohibited 11am-4pm, cooling centers active."
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Exported NDMA HAP triggers for {len(city_records)} cities to {json_path} and {parquet_path}.")
    return manifest


if __name__ == "__main__":
    build_ndma_hap_triggers()
