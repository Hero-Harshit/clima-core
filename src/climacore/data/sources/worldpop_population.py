"""
WorldPop 2024 High-Resolution (100m) Population Grid Extraction & Human Exposure Engine.

Extracts multi-scale urban concentric exposure buffers (5km core, 15km metro, 30km regional),
calculates population densities, identifies informal high-density vulnerability clusters,
and establishes population-weighted disaster impact baselines for all 50 Indian cities.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

from climacore.common.logging import get_logger
from climacore.common.provenance import compute_sha256
from climacore.config.loader import get_project_root

logger = get_logger("climacore.data.sources.worldpop_population")


def extract_city_population_exposure(
    tif_path: Path,
    cities_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Extract concentric population counts and densities for all cities using windowed raster reads."""
    results: List[Dict[str, Any]] = []
    t0 = time.time()

    with rasterio.open(tif_path) as src:
        nodata = src.nodata if src.nodata is not None else -99999.0
        transform = src.transform

        logger.info(f"Opened WorldPop 2024 GeoTIFF: Shape={src.shape}, CRS={src.crs}, Resolution={src.res}.")

        for _, row in cities_df.iterrows():
            city = str(row["city"]).strip()
            state = str(row["state"]).strip() if pd.notna(row.get("state")) else "India"
            lat = float(row["latitude"])
            lon = float(row["longitude"])

            # 3 concentric radii in kilometers
            city_metrics: Dict[str, Any] = {
                "city": city,
                "state": state,
                "latitude": lat,
                "longitude": lon,
            }

            for r_km in [5.0, 15.0, 30.0]:
                dlat = r_km / 110.574
                dlon = r_km / (111.320 * np.cos(np.radians(lat)))

                min_x = lon - dlon
                max_x = lon + dlon
                min_y = lat - dlat
                max_y = lat + dlat

                window = from_bounds(min_x, min_y, max_x, max_y, transform)
                arr = src.read(1, window=window)

                # Clean nodata and negatives
                clean_arr = np.where((arr >= 0.0) & (arr != nodata), arr, 0.0)

                # Circular mask within radius
                height, width = clean_arr.shape
                y_coords, x_coords = np.mgrid[0:height, 0:width]
                
                # Transform pixel indices back to spatial coords
                xs, ys = rasterio.transform.xy(transform, window.row_off + y_coords, window.col_off + x_coords)
                xs = np.array(xs)
                ys = np.array(ys)

                # Approximate distance in km from city center
                dy_km = (ys - lat) * 110.574
                dx_km = (xs - lon) * (111.320 * np.cos(np.radians(lat)))
                dist_km = np.sqrt(dx_km**2 + dy_km**2)

                mask = dist_km <= r_km
                pop_in_circle = float(np.sum(clean_arr[mask]))
                area_km2 = np.pi * (r_km**2)
                density = pop_in_circle / area_km2

                r_str = f"{int(r_km)}km"
                city_metrics[f"population_{r_str}"] = int(round(pop_in_circle))
                city_metrics[f"density_{r_str}_per_km2"] = round(density, 1)

                if int(r_km) == 15:
                    # High-density cluster exposure (cells with >= 200 people per 100m x 100m = >= 20,000/km2)
                    dense_mask = mask & (clean_arr >= 200.0)
                    dense_pop = float(np.sum(clean_arr[dense_mask]))
                    peak_cell = float(np.max(clean_arr[mask])) if np.any(mask) else 0.0

                    city_metrics["dense_cluster_pop_15km"] = int(round(dense_pop))
                    city_metrics["peak_cell_population_100m"] = int(round(peak_cell))

            # Assign Urban Vulnerability Tier
            pop_15k = city_metrics["population_15km"]
            density_5k = city_metrics["density_5km_per_km2"]

            if pop_15k >= 8_000_000 or density_5k >= 15_000:
                tier = "EXTREME_METROPOLITAN_DENSITY"
            elif pop_15k >= 3_000_000 or density_5k >= 8_000:
                tier = "HIGH_URBAN_EXPOSURE"
            elif pop_15k >= 1_000_000:
                tier = "MODERATE_TIER2_EXPOSURE"
            else:
                tier = "TIER3_EMERGING_URBAN"

            city_metrics["urban_exposure_tier"] = tier
            results.append(city_metrics)

    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values(by="population_15km", ascending=False).reset_index(drop=True)

    elapsed = time.time() - t0
    logger.info(f"Processed 50 cities in {elapsed:.2f}s. Total population evaluated in 30km: {df_out['population_30km'].sum():,}.")
    return df_out, {"elapsed_seconds": round(elapsed, 2), "total_30km_population": int(df_out["population_30km"].sum())}


def process_worldpop_pipeline() -> Tuple[pd.DataFrame, Path, Path]:
    """Run full ingestion, save Parquet & JSON, and generate provenance manifest."""
    root = get_project_root()
    tif_path = root / "data" / "raw" / "population" / "ind_pop_2024_CN_100m_R2024B_v1.tif"
    cities_file = root / "data" / "raw" / "india_weather_air_quality" / "cities.csv"
    proc_dir = root / "data" / "processed" / "population"
    proc_dir.mkdir(parents=True, exist_ok=True)

    if not tif_path.exists():
        raise FileNotFoundError(f"WorldPop GeoTIFF missing at {tif_path}")

    cities_df = pd.read_csv(cities_file)
    logger.info(f"Starting WorldPop 100m ingestion for {len(cities_df)} Indian cities...")

    df_pop, meta = extract_city_population_exposure(tif_path, cities_df)

    parquet_path = proc_dir / "city_population_exposure.parquet"
    json_path = proc_dir / "city_population_exposure.json"

    df_pop.to_parquet(parquet_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df_pop.to_dict(orient="records"), f, indent=2)

    # Manifest
    manifest_path = root / "data" / "manifests" / "worldpop_manifest.json"
    manifest = {
        "source_id": "worldpop_india_2024_100m_constrained",
        "display_name": "WorldPop India 2024 Constrained 100m Population Grid (R2024B_v1)",
        "provider": "WorldPop / University of Southampton (CC-BY 4.0)",
        "raw_file": tif_path.name,
        "spatial_resolution": "0.000833 degrees (~100m)",
        "crs": "EPSG:4326 (WGS84)",
        "cities_evaluated": len(df_pop),
        "total_30km_population": meta["total_30km_population"],
        "sha256_raw": compute_sha256(tif_path),
        "sha256_parquet": compute_sha256(parquet_path),
        "sha256_json": compute_sha256(json_path),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "buffers_analyzed": {
            "5km": "Inner high-density municipal core",
            "15km": "Primary metropolitan and peri-urban footprint",
            "30km": "Greater urban commuting region and district basin"
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"WorldPop extraction completed. Manifest saved to {manifest_path}.")
    return df_pop, parquet_path, json_path


if __name__ == "__main__":
    process_worldpop_pipeline()

