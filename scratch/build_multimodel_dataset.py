"""
Fetch and align multi-model NWP previous runs (GFS, JMA, ICON, BestMatch) with IMD ground truth observations.
Cities: Delhi, Mumbai, Kolkata, Nagpur (2024-01-20 to 2024-11-15).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "forecast_correction_pairs"

CITIES = {
    "delhi": {"lat": 28.6139, "lon": 77.2090, "elevation": 216.0, "zone": "composite_north"},
    "mumbai": {"lat": 19.0760, "lon": 72.8777, "elevation": 14.0, "zone": "warm_humid_west"},
    "kolkata": {"lat": 22.5726, "lon": 88.3639, "elevation": 9.0, "zone": "warm_humid_east"},
    "nagpur": {"lat": 21.1458, "lon": 79.0882, "elevation": 310.0, "zone": "hot_dry_central"},
}


def main():
    logger.info("Loading existing paired IMD verification observations...")
    existing_df = pd.read_parquet(DATA_DIR / "daily_forecast_obs_pairs.parquet")
    logger.info("Existing records: %d", len(existing_df))

    records = []
    start_date = "2024-01-20"
    end_date = "2024-11-15"

    for city_name, meta in CITIES.items():
        lat = meta["lat"]
        lon = meta["lon"]
        logger.info("Fetching multi-model daily forecasts for %s...", city_name)
        url = (
            f"https://previous-runs-api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&"
            f"daily=temperature_2m_max,temperature_2m_min,precipitation_sum&"
            f"models=gfs_seamless,jma_seamless,icon_seamless"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "ClimaCore/1.3"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        d = data["daily"]
        city_df = pd.DataFrame({
            "city": city_name,
            "date": pd.to_datetime(d["time"]),
            "tmax_gfs": d["temperature_2m_max_gfs_seamless"],
            "tmin_gfs": d["temperature_2m_min_gfs_seamless"],
            "precip_gfs": d["precipitation_sum_gfs_seamless"],
            "tmax_jma": d["temperature_2m_max_jma_seamless"],
            "tmin_jma": d["temperature_2m_min_jma_seamless"],
            "precip_jma": d["precipitation_sum_jma_seamless"],
            "tmax_icon": d["temperature_2m_max_icon_seamless"],
            "tmin_icon": d["temperature_2m_min_icon_seamless"],
            "precip_icon": d["precipitation_sum_icon_seamless"],
        })
        records.append(city_df)
        time.sleep(0.5)

    multi_df = pd.concat(records, ignore_index=True)
    logger.info("Multi-model records fetched: %d", len(multi_df))

    # Merge with existing verifying IMD observations
    merged = pd.merge(
        existing_df,
        multi_df,
        on=["city", "date"],
        how="inner"
    )
    logger.info("Merged verified multi-model records: %d", len(merged))

    out_path = DATA_DIR / "multimodel_forecast_obs_pairs.parquet"
    merged.to_parquet(out_path, index=False)
    logger.info("Saved multi-model verification dataset to %s", out_path)


if __name__ == "__main__":
    main()
