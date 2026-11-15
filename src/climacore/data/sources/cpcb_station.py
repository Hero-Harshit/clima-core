"""
Source adapter for Central Pollution Control Board (CPCB) physical ground-monitoring stations.
Cleans, standardizes, and exports physical monitoring station ground-truth observations.
"""

from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime, timezone
import pandas as pd

from climacore.common.logging import get_logger
from climacore.common.provenance import compute_sha256, get_file_metadata
from climacore.config.loader import get_project_root

logger = get_logger("climacore.data.sources.cpcb_station")


def process_cpcb_station_data(
    raw_dir: Path | str = None,
    output_dir: Path | str = None,
) -> pd.DataFrame:
    """Standardize CPCB physical station observations."""
    root = get_project_root()
    raw_path = Path(raw_dir) if raw_dir else root / "data" / "raw" / "cpcb_station_data"
    out_path = Path(output_dir) if output_dir else root / "data" / "processed"
    out_path.mkdir(parents=True, exist_ok=True)

    city_day_file = raw_path / "city_day.csv"
    if not city_day_file.exists():
        raise FileNotFoundError(f"Missing CPCB file: {city_day_file}")

    logger.info("Loading and processing CPCB city_day.csv...")
    df = pd.read_csv(city_day_file)

    # Standardize column names
    rename_map = {
        "City": "city",
        "Date": "date",
        "PM2.5": "cpcb_pm2_5_mean",
        "PM10": "cpcb_pm10_mean",
        "NO2": "cpcb_no2_mean",
        "SO2": "cpcb_so2_mean",
        "CO": "cpcb_co_mean",
        "O3": "cpcb_o3_mean",
        "AQI": "cpcb_aqi",
        "AQI_Bucket": "cpcb_aqi_category",
    }
    df = df.rename(columns=rename_map)

    # Filter columns
    keep_cols = [c for c in rename_map.values() if c in df.columns]
    df_clean = df[keep_cols].copy()
    df_clean["date"] = pd.to_datetime(df_clean["date"]).dt.strftime("%Y-%m-%d")

    # Drop rows where both PM2.5 and AQI are null
    df_clean = df_clean[df_clean["cpcb_pm2_5_mean"].notnull() | df_clean["cpcb_aqi"].notnull()].copy()
    df_clean = df_clean.sort_values(by=["city", "date"]).reset_index(drop=True)

    # Save to Parquet
    out_file = out_path / "cpcb_ground_truth_cities.parquet"
    df_clean.to_parquet(out_file, index=False)
    logger.info(f"Saved cleaned CPCB physical station observations to {out_file}: {len(df_clean):,} rows.")

    # Manifest
    man_path = root / "data" / "manifests" / "cpcb_station_manifest.json"
    manifest_data = {
        "source_id": "cpcb_ground_truth",
        "display_name": "CPCB CAAQMS Physical Ground Station Air Quality Data",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "Central Pollution Control Board (CPCB) via official station records",
        "spatial_resolution": "Physical CAAQMS ground monitoring stations",
        "total_records": len(df_clean),
        "cities_count": int(df_clean["city"].nunique()),
        "start_date": df_clean["date"].min(),
        "end_date": df_clean["date"].max(),
        "notes": "Direct physical continuous ambient air quality monitoring stations (CAAQMS).",
    }
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return df_clean


if __name__ == "__main__":
    process_cpcb_station_data()
