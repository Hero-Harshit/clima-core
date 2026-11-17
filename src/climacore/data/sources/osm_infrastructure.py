"""
OpenStreetMap (OSM) Critical Infrastructure & Population-Weighted Hazard Exposure Pipeline.
Extracts hospitals, clinics, schools, bridges, transit hubs, drainage canals, arterial roads,
and industrial zones within 5km municipal core and 15km metropolitan radius buffers across 50 Indian cities.
Stream-parsed with C++ pyosmium FileProcessor to ensure < 250 MB memory usage on 8 GB RAM systems.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import osmium
from osmium.filter import EmptyTagFilter, KeyFilter

from climacore.common.logging import get_logger
from climacore.config.loader import get_project_root

logger = get_logger("climacore.data.sources.osm_infrastructure")

# Zone to States mapping in Geofabrik India sub-extracts
ZONE_STATE_MAP: Dict[str, List[str]] = {
    "central-zone-261116.osm.pbf": [
        "Madhya Pradesh", "Chhattisgarh", "Uttar Pradesh", "Uttarakhand"
    ],
    "eastern-zone-261116.osm.pbf": ["Bihar", "Jharkhand", "Odisha", "West Bengal", "Andaman and Nicobar Islands"],
    "north-eastern-zone-261116.osm.pbf": ["Assam", "Manipur", "Meghalaya", "Sikkim", "Tripura"],
    "northern-zone-261116.osm.pbf": [
        "Chandigarh", "Delhi", "Himachal Pradesh", "Jammu and Kashmir",
        "Ladakh", "Punjab", "Rajasthan"
    ],
    "southern-zone-261116.osm.pbf": [
        "Andhra Pradesh", "Karnataka", "Kerala", "Puducherry", "Tamil Nadu",
        "Telangana", "Andaman and Nicobar Islands"
    ],
    "western-zone-261116.osm.pbf": ["Goa", "Gujarat", "Maharashtra"]
}

DEFAULT_METRICS = {
    "hospitals_5km": 0,
    "clinics_5km": 0,
    "schools_5km": 0,
    "bridges_5km": 0,
    "drainage_canals_5km": 0,
    "arterial_roads_5km": 0,
    "hospitals_15km": 0,
    "clinics_15km": 0,
    "schools_15km": 0,
    "bridges_15km": 0,
    "drainage_canals_15km": 0,
    "arterial_roads_15km": 0,
    "industrial_zones_15km": 0,
    "transit_stations_15km": 0,
}


def _sha256(path: Path) -> str:
    """Compute SHA-256 checksum for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1048576):
            h.update(chunk)
    return h.hexdigest()


def extract_zone_infrastructure(
    pbf_path: Path,
    city_tuples: List[Tuple[str, float, float, float]]
) -> Dict[str, Dict[str, int]]:
    """
    Stream-parse an OSM PBF file and extract counts for assigned cities.
    city_tuples: [(city_name, lat, lon, cos_lat), ...]
    """
    counts: Dict[str, Dict[str, int]] = {
        c[0]: dict(DEFAULT_METRICS) for c in city_tuples
    }

    if not city_tuples:
        return counts

    # Pre-calculate bounding box encompassing all assigned cities in this zone (+ 0.16 deg buffer)
    min_lat = min(c[1] for c in city_tuples) - 0.15
    max_lat = max(c[1] for c in city_tuples) + 0.15
    min_lon = min(c[2] for c in city_tuples) - 0.17
    max_lon = max(c[2] for c in city_tuples) + 0.17

    logger.info(f"Streaming {pbf_path.name} for {len(city_tuples)} cities...")
    t0 = time.time()

    fp = (
        osmium.FileProcessor(str(pbf_path))
        .with_locations()
        .with_filter(EmptyTagFilter())
        .with_filter(KeyFilter(
            "amenity", "highway", "waterway", "landuse", "railway", "aeroway", "bridge"
        ))
    )

    item_count = 0
    match_count = 0

    for obj in fp:
        item_count += 1
        tags = obj.tags

        amenity = tags.get("amenity")
        bridge = tags.get("bridge")
        highway = tags.get("highway")
        waterway = tags.get("waterway")
        landuse = tags.get("landuse")
        railway = tags.get("railway")
        aeroway = tags.get("aeroway")

        # Resolve representative coordinate
        if obj.is_node():
            loc = obj.location
            if not loc.valid():
                continue
            lat, lon = loc.lat, loc.lon
        elif obj.is_way() and len(obj.nodes) > 0 and obj.nodes[0].location.valid():
            lat, lon = obj.nodes[0].lat, obj.nodes[0].lon
        else:
            continue

        # Fast zone-wide bounding check
        if lat < min_lat or lat > max_lat or lon < min_lon or lon > max_lon:
            continue

        # Check against assigned cities
        for c_name, c_lat, c_lon, cos_lat in city_tuples:
            if abs(lat - c_lat) > 0.14 or abs(lon - c_lon) > 0.16:
                continue

            dy = (lat - c_lat) * 110.574
            dx = (lon - c_lon) * 111.320 * cos_lat
            d2 = dy * dy + dx * dx

            if d2 > 225.0:  # Beyond 15km squared
                continue

            match_count += 1
            m = counts[c_name]
            is_5km = d2 <= 25.0  # Within 5km squared

            # 1. Healthcare
            if amenity == "hospital":
                m["hospitals_15km"] += 1
                if is_5km:
                    m["hospitals_5km"] += 1
            elif amenity in ["clinic", "doctors", "health_post", "pharmacy"]:
                m["clinics_15km"] += 1
                if is_5km:
                    m["clinics_5km"] += 1

            # 2. Education
            if amenity in ["school", "college", "university", "kindergarten"]:
                m["schools_15km"] += 1
                if is_5km:
                    m["schools_5km"] += 1

            # 3. Bridges & Flyovers
            if bridge == "yes":
                m["bridges_15km"] += 1
                if is_5km:
                    m["bridges_5km"] += 1

            # 4. Drainage & Waterways
            if waterway in ["canal", "drain", "stream", "ditch", "river"]:
                m["drainage_canals_15km"] += 1
                if is_5km:
                    m["drainage_canals_5km"] += 1

            # 5. Arterial Roads
            if highway in ["motorway", "trunk", "primary", "motorway_link", "trunk_link", "primary_link"]:
                m["arterial_roads_15km"] += 1
                if is_5km:
                    m["arterial_roads_5km"] += 1

            # 6. Industrial zones
            if landuse == "industrial":
                m["industrial_zones_15km"] += 1

            # 7. Transit hubs
            if railway in ["station", "halt"] or aeroway in ["aerodrome", "terminal"]:
                m["transit_stations_15km"] += 1

    dt = time.time() - t0
    logger.info(
        f"Completed {pbf_path.name} in {dt:.2f}s: scanned {item_count:,} items, matched {match_count:,} features."
    )
    return counts


def process_osm_pipeline() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run full OSM extraction across all 6 Geofabrik zones and merge with WorldPop exposure."""
    root = get_project_root()
    osm_dir = root / "data" / "raw" / "osm"
    pop_path = root / "data" / "processed" / "population" / "city_population_exposure.parquet"
    out_dir = root / "data" / "processed" / "infrastructure"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = root / "data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    if not pop_path.exists():
        raise FileNotFoundError(f"Population dataset not found at {pop_path}")

    pop_df = pd.read_parquet(pop_path)
    city_master = pop_df[["city", "state", "latitude", "longitude", "population_15km", "dense_cluster_pop_15km"]].copy()

    total_infra_counts: Dict[str, Dict[str, int]] = {
        row["city"]: dict(DEFAULT_METRICS) for _, row in city_master.iterrows()
    }

    raw_manifest_checksums: Dict[str, str] = {}

    start_total = time.time()
    logger.info(f"Starting OSM Critical Infrastructure extraction for 50 cities from {osm_dir}...")

    # Iterate over available zone files
    for pbf_filename, assigned_states in ZONE_STATE_MAP.items():
        pbf_file = osm_dir / pbf_filename
        if not pbf_file.exists():
            logger.warning(f"OSM PBF file not found: {pbf_file}, skipping...")
            continue

        raw_manifest_checksums[pbf_filename] = _sha256(pbf_file)

        # Filter cities belonging to this zone
        zone_cities = city_master[city_master["state"].isin(assigned_states)]
        city_tuples = [
            (
                row["city"],
                float(row["latitude"]),
                float(row["longitude"]),
                float(np.cos(np.radians(row["latitude"])))
            )
            for _, row in zone_cities.iterrows()
        ]

        zone_counts = extract_zone_infrastructure(pbf_file, city_tuples)

        # Merge counts into master dictionary (handles multi-zone cities like Port Blair seamlessly)
        for city_name, metrics in zone_counts.items():
            for k, val in metrics.items():
                total_infra_counts[city_name][k] += val

    # Combine with Population and calculate derived ratios
    results: List[Dict[str, Any]] = []
    area_15km_km2 = np.pi * (15.0 ** 2)

    for _, row in city_master.iterrows():
        city = row["city"]
        counts = total_infra_counts[city]
        pop_15k = int(row["population_15km"])

        hospitals_15k = counts["hospitals_15km"]
        clinics_15k = counts["clinics_15km"]
        total_health_15k = hospitals_15k + clinics_15k
        schools_15k = counts["schools_15km"]
        bridges_15k = counts["bridges_15km"]
        drains_15k = counts["drainage_canals_15km"]
        arterial_15k = counts["arterial_roads_15km"]
        industrial_15k = counts["industrial_zones_15km"]
        transit_15k = counts["transit_stations_15km"]

        # Ratios
        people_per_hospital = int(round(pop_15k / max(1, hospitals_15k)))
        people_per_school = int(round(pop_15k / max(1, schools_15k)))
        infra_density = round(float((total_health_15k + schools_15k + bridges_15k) / area_15km_km2), 2)
        drainage_artery_ratio = round(float(drains_15k / max(1, arterial_15k)), 3)

        res_record = {
            "city": city,
            "state": row["state"],
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "population_15km": pop_15k,
            "dense_cluster_pop_15km": int(row["dense_cluster_pop_15km"]),
            # 5 km metrics
            "hospitals_5km": counts["hospitals_5km"],
            "clinics_5km": counts["clinics_5km"],
            "schools_5km": counts["schools_5km"],
            "bridges_5km": counts["bridges_5km"],
            "drainage_canals_5km": counts["drainage_canals_5km"],
            "arterial_roads_5km": counts["arterial_roads_5km"],
            # 15 km metrics
            "hospitals_15km": hospitals_15k,
            "clinics_15km": clinics_15k,
            "total_healthcare_15km": total_health_15k,
            "schools_15km": schools_15k,
            "bridges_15km": bridges_15k,
            "drainage_canals_15km": drains_15k,
            "arterial_roads_15km": arterial_15k,
            "industrial_zones_15km": industrial_15k,
            "transit_stations_15km": transit_15k,
            # Population-Weighted Exposure Ratios
            "people_per_hospital_15km": people_per_hospital,
            "people_per_school_15km": people_per_school,
            "critical_infrastructure_density_km2": infra_density,
            "drainage_arterial_ratio": drainage_artery_ratio,
        }
        results.append(res_record)

    df_out = pd.DataFrame(results)

    # Save to Parquet and JSON
    parquet_path = out_dir / "city_infrastructure_exposure.parquet"
    json_path = out_dir / "city_infrastructure_exposure.json"

    df_out.to_parquet(parquet_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - start_total
    logger.info(f"Processed infrastructure exposure for 50 cities in {total_time:.2f}s.")

    # Manifest metadata
    manifest = {
        "source_id": "openstreetmap_geofabrik_india_infrastructure",
        "display_name": "OpenStreetMap Geofabrik India Critical Infrastructure Grids",
        "provider": "OpenStreetMap contributors / Geofabrik GmbH (ODbL 1.0)",
        "cities_evaluated": len(df_out),
        "total_hospitals_15km": int(df_out["hospitals_15km"].sum()),
        "total_schools_15km": int(df_out["schools_15km"].sum()),
        "total_bridges_15km": int(df_out["bridges_15km"].sum()),
        "total_drainage_canals_15km": int(df_out["drainage_canals_15km"].sum()),
        "raw_zone_checksums": raw_manifest_checksums,
        "sha256_parquet": _sha256(parquet_path),
        "sha256_json": _sha256(json_path),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "buffers_analyzed": {
            "5km": "Municipal core critical social and lifeline infrastructure",
            "15km": "Metropolitan disaster exposure footprint, transit networks, and drainage"
        }
    }

    manifest_path = manifest_dir / "osm_infrastructure_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"OSM manifest saved to {manifest_path}.")
    return df_out, manifest


if __name__ == "__main__":
    process_osm_pipeline()
