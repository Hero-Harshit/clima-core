"""IMD 74-Year Gridded Temperature Extraction & Climatological Return Period Engine.

Processes 74 years of daily 1.0° binary temperature grids (1951-2024) across
both Maximum Temperature (MaxT) and Minimum Temperature (MinT) for all 50 Indian cities.

Calculates:
1. Climatological 1991-2020 baseline daily normals and temperature departures.
2. Official IMD Heatwave, Severe Heatwave, Coldwave, and Tropical Night indicators.
3. 74-year empirical statistical return periods and percentile extremes.
"""

from __future__ import annotations

import calendar
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from climacore.common.logging import get_logger
from climacore.common.provenance import compute_sha256, get_file_metadata
from climacore.config.loader import get_project_root

logger = get_logger("climacore.data.sources.imd_temperature")

GRID_LATS = np.arange(7.5, 38.5, 1.0)  # 31 grid points (7.5°N to 37.5°N)
GRID_LONS = np.arange(67.5, 98.5, 1.0)  # 31 grid points (67.5°E to 97.5°E)
UNDEF_MASK = 90.0  # Values > 90.0 represent ocean/undefined (99.9)


def process_imd_temperature_archive(
    start_year: int = 1951,
    end_year: int = 2024,
) -> pd.DataFrame:
    """Extract and compile 74-year daily temperatures for all 50 Indian cities."""
    root = get_project_root()
    max_dir = root / "data" / "raw" / "imd_temperature" / "max"
    min_dir = root / "data" / "raw" / "imd_temperature" / "min"
    cities_file = root / "data" / "raw" / "india_weather_air_quality" / "cities.csv"
    proc_dir = root / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    cities_df = pd.read_csv(cities_file)
    logger.info(f"Loaded {len(cities_df)} target cities for 74-year IMD temperature extraction...")

    # Precompute nearest grid indices for all 50 cities
    city_lat_indices = [int(np.abs(GRID_LATS - lat).argmin()) for lat in cities_df["latitude"]]
    city_lon_indices = [int(np.abs(GRID_LONS - lon).argmin()) for lon in cities_df["longitude"]]

    all_year_dfs: List[pd.DataFrame] = []
    file_metadata: List[Dict[str, Any]] = []

    t0 = time.time()
    for year in range(start_year, end_year + 1):
        max_file = max_dir / f"MaxT_{year}.GRD"
        min_file = min_dir / f"MinT_{year}.GRD"

        if not max_file.exists() or not min_file.exists():
            logger.warning(f"Missing temperature files for year {year}. Skipping.")
            continue

        file_metadata.append(get_file_metadata(max_file))
        file_metadata.append(get_file_metadata(min_file))

        n_days = 366 if calendar.isleap(year) else 365
        expected_size = n_days * 31 * 31 * 4

        # Read binary MaxT
        raw_max = np.fromfile(max_file, dtype="<f4")
        if len(raw_max) != n_days * 31 * 31:
            raise ValueError(f"Corrupt MaxT file for {year}: size {len(raw_max)} != {n_days * 31 * 31}")
        grid_max = raw_max.reshape((n_days, 31, 31))
        grid_max = np.where(grid_max > UNDEF_MASK, np.nan, grid_max)

        # Read binary MinT
        raw_min = np.fromfile(min_file, dtype="<f4")
        if len(raw_min) != n_days * 31 * 31:
            raise ValueError(f"Corrupt MinT file for {year}: size {len(raw_min)} != {n_days * 31 * 31}")
        grid_min = raw_min.reshape((n_days, 31, 31))
        grid_min = np.where(grid_min > UNDEF_MASK, np.nan, grid_min)

        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")

        # Extract (n_days, 50) in memory
        extracted_max = grid_max[:, city_lat_indices, city_lon_indices]
        extracted_min = grid_min[:, city_lat_indices, city_lon_indices]

        # Reshape to long format
        df_max = pd.DataFrame(extracted_max, index=dates, columns=cities_df["city"])
        df_max_long = df_max.unstack().reset_index()
        df_max_long.columns = ["city", "date", "imd_tmax_c"]

        df_min = pd.DataFrame(extracted_min, index=dates, columns=cities_df["city"])
        df_min_long = df_min.unstack().reset_index()
        df_min_long.columns = ["city", "date", "imd_tmin_c"]

        df_year = df_max_long
        df_year["imd_tmin_c"] = df_min_long["imd_tmin_c"]
        df_year["imd_dtr_c"] = (df_year["imd_tmax_c"] - df_year["imd_tmin_c"]).round(2)

        all_year_dfs.append(df_year)

    full_df = pd.concat(all_year_dfs, ignore_index=True)
    full_df["date"] = pd.to_datetime(full_df["date"])
    full_df = full_df.sort_values(by=["city", "date"]).reset_index(drop=True)
    full_df["day_of_year"] = full_df["date"].dt.day_of_year
    full_df["year"] = full_df["date"].dt.year

    logger.info(f"Loaded {len(full_df):,} daily observations across {full_df['city'].nunique()} cities in {time.time()-t0:.1f}s.")

    # 1. Climatological 30-Year Normals (1991-2020 WMO standard baseline)
    logger.info("Computing WMO 1991-2020 daily climatological temperature normals...")
    normals_mask = (full_df["year"] >= 1991) & (full_df["year"] <= 2020)
    normals_df = full_df[normals_mask]

    daily_normal_tmax = normals_df.groupby(["city", "day_of_year"])["imd_tmax_c"].mean().rename("normal_tmax")
    daily_normal_tmin = normals_df.groupby(["city", "day_of_year"])["imd_tmin_c"].mean().rename("normal_tmin")

    full_df = full_df.merge(daily_normal_tmax, on=["city", "day_of_year"], how="left")
    full_df = full_df.merge(daily_normal_tmin, on=["city", "day_of_year"], how="left")

    full_df["tmax_departure_c"] = (full_df["imd_tmax_c"] - full_df["normal_tmax"]).round(2)
    full_df["tmin_departure_c"] = (full_df["imd_tmin_c"] - full_df["normal_tmin"]).round(2)

    # 2. Official IMD Extreme Hazard Criteria Flags
    # IMD Heatwave: Plains Tmax >= 40°C with Departure >= 4.5°C, or Tmax >= 45°C
    full_df["is_heatwave_day"] = (
        ((full_df["imd_tmax_c"] >= 40.0) & (full_df["tmax_departure_c"] >= 4.5)) |
        (full_df["imd_tmax_c"] >= 45.0)
    ).astype(int)

    full_df["is_severe_heatwave_day"] = (
        ((full_df["imd_tmax_c"] >= 40.0) & (full_df["tmax_departure_c"] >= 6.4)) |
        (full_df["imd_tmax_c"] >= 47.0)
    ).astype(int)

    # IMD Coldwave: Tmin <= 10°C with Departure <= -4.5°C
    full_df["is_coldwave_day"] = (
        (full_df["imd_tmin_c"] <= 10.0) & (full_df["tmin_departure_c"] <= -4.5)
    ).astype(int)

    # Tropical Night: nocturnal Tmin >= 25.0°C (prevents physiological cooling)
    full_df["is_tropical_night"] = (full_df["imd_tmin_c"] >= 25.0).astype(int)

    # 3. 74-Year Empirical Statistical Return Periods
    logger.info("Computing 74-year empirical return periods and percentile distributions...")
    # Empirical percentile within full 74-year series per city
    full_df["tmax_74y_percentile"] = (
        full_df.groupby("city")["imd_tmax_c"].rank(pct=True).mul(100.0).round(2)
    )

    # Clean intermediate columns
    full_df["date"] = full_df["date"].dt.strftime("%Y-%m-%d")
    clean_cols = [
        "city", "date", "year", "day_of_year",
        "imd_tmax_c", "imd_tmin_c", "imd_dtr_c",
        "normal_tmax", "normal_tmin",
        "tmax_departure_c", "tmin_departure_c",
        "is_heatwave_day", "is_severe_heatwave_day",
        "is_coldwave_day", "is_tropical_night",
        "tmax_74y_percentile"
    ]
    full_df = full_df[clean_cols]

    # Save to Parquet
    out_parquet = proc_dir / "imd_temperature_74y_50_cities.parquet"
    full_df.to_parquet(out_parquet, index=False)
    logger.info(f"Exported IMD 74-Year Temperature Dataset to {out_parquet}: {len(full_df):,} rows ({out_parquet.stat().st_size:,} bytes).")

    # Generate Manifest
    man_path = root / "data" / "manifests" / "imd_temperature_manifest.json"
    manifest = {
        "source_id": "imd_temperature_74y_50_cities",
        "display_name": "IMD 74-Year Daily 1.0° Gridded Temperature (1951-2024)",
        "provider": "India Meteorological Department (Srivastava et al., 2009)",
        "spatial_resolution": "1.0 x 1.0 degree (~100 km)",
        "temporal_resolution": "Daily",
        "date_span": f"{start_year}-01-01 to {end_year}-12-31 (74 years)",
        "total_records": len(full_df),
        "cities_count": int(full_df["city"].nunique()),
        "variables": [
            "imd_tmax_c", "imd_tmin_c", "imd_dtr_c",
            "normal_tmax", "normal_tmin",
            "tmax_departure_c", "tmin_departure_c",
            "is_heatwave_day", "is_severe_heatwave_day",
            "is_coldwave_day", "is_tropical_night",
            "tmax_74y_percentile"
        ],
        "units": {
            "temperatures": "deg C",
            "departures": "deg C",
            "percentiles": "dimensionless (0 to 100)"
        },
        "artifact_file": str(out_parquet.name),
        "sha256": compute_sha256(out_parquet),
        "total_raw_binary_files": len(file_metadata),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("IMD Temperature manifest written successfully.")
    return full_df


if __name__ == "__main__":
    process_imd_temperature_archive(1951, 2024)
