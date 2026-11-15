"""
Source adapter for IMD 0.25° Daily Gridded Rainfall NetCDF dataset.
Reads NetCDF files, extracts physical rain gauge observations for the 50 Indian cities,
cleans missing/invalid values, and exports a standardized Parquet table.
"""

from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime, timezone
import xarray as xr
import pandas as pd
import numpy as np

from climacore.common.logging import get_logger
from climacore.common.provenance import compute_sha256, get_file_metadata
from climacore.config.loader import get_project_root

logger = get_logger("climacore.data.sources.imd_rainfall")


def extract_imd_rainfall_for_cities(
    raw_dir: Path | str = None,
    cities_file: Path | str = None,
    output_dir: Path | str = None,
) -> pd.DataFrame:
    """Extract daily rainfall series for all 50 cities from IMD NetCDF files."""
    root = get_project_root()
    raw_path = Path(raw_dir) if raw_dir else root / "data" / "raw" / "imd_rainfall"
    cities_path = Path(cities_file) if cities_file else root / "data" / "raw" / "india_weather_air_quality" / "cities.csv"
    out_path = Path(output_dir) if output_dir else root / "data" / "processed"
    out_path.mkdir(parents=True, exist_ok=True)

    nc_files = sorted(list(raw_path.glob("*.nc")))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found in {raw_path}")

    logger.info(f"Found {len(nc_files)} IMD NetCDF files in {raw_path}.")
    df_cities = pd.read_csv(cities_path)

    all_records = []
    file_metadata = []

    for nc_file in nc_files:
        logger.info(f"Processing {nc_file.name}...")
        meta = get_file_metadata(nc_file)
        file_metadata.append(meta)

        ds = xr.open_dataset(nc_file)
        # Lat/Lon variable naming check
        lat_var = "LATITUDE" if "LATITUDE" in ds else "lat"
        lon_var = "LONGITUDE" if "LONGITUDE" in ds else "lon"
        time_var = "TIME" if "TIME" in ds else "time"
        rain_var = "RAINFALL" if "RAINFALL" in ds else list(ds.data_vars.keys())[0]

        # Extract for each city using nearest neighbor
        for _, c_row in df_cities.iterrows():
            city_name = c_row["city"]
            c_lat = float(c_row["latitude"])
            c_lon = float(c_row["longitude"])

            sub = ds[rain_var].sel({lat_var: c_lat, lon_var: c_lon}, method="nearest")
            time_vals = pd.to_datetime(sub[time_var].values)
            rain_vals = sub.values

            # Replace missing / negative fill values (IMD uses -999.0 for invalid/ocean)
            rain_vals = np.where(rain_vals < 0, np.nan, rain_vals)

            city_df = pd.DataFrame({
                "city": city_name,
                "date": time_vals.strftime("%Y-%m-%d"),
                "imd_rainfall_mm": np.round(rain_vals, 2),
            })
            all_records.append(city_df)

        ds.close()

    combined_df = pd.concat(all_records, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["city", "date"]).sort_values(by=["city", "date"]).reset_index(drop=True)

    # Save to Parquet
    out_file = out_path / "imd_rainfall_50_cities.parquet"
    combined_df.to_parquet(out_file, index=False)
    logger.info(f"Saved extracted IMD rainfall data to {out_file}: {len(combined_df):,} rows.")

    # Generate Manifest
    man_path = root / "data" / "manifests" / "imd_rainfall_manifest.json"
    manifest_data = {
        "source_id": "imd_rainfall",
        "display_name": "IMD 0.25° Daily Gridded Rainfall",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "India Meteorological Department (Pai et al., 2014)",
        "spatial_resolution": "0.25 x 0.25 degree (~25 km)",
        "files": file_metadata,
        "extracted_cities_count": len(df_cities),
        "total_records": len(combined_df),
        "start_date": combined_df["date"].min(),
        "end_date": combined_df["date"].max(),
        "notes": "Physical rain gauge ground truth interpolated onto regular grid. Negative fill values cleaned to NaN.",
    }
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return combined_df


if __name__ == "__main__":
    extract_imd_rainfall_for_cities()
