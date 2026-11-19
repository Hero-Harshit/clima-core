"""Build matched historical forecast-observation pairs dataset from Open-Meteo Previous Runs API.

Fetches matched runs (verifying day-0 analysis vs day-1 lead forecast)
across 4 representative climate anchor cities in India:
- Delhi (North / Semiarid / extreme summer heatwave & winter fog)
- Mumbai (West Coast / Tropical wet & dry / heavy monsoon)
- Kolkata (East / Tropical wet / humid monsoon)
- Nagpur (Central / Dry continental summer)

Covers all 4 seasons (Winter, Summer Heatwave, Southwest Monsoon, Post-Monsoon)
with zero missing values for all paired variables:
- temperature_2m vs temperature_2m_previous_day1
- precipitation vs precipitation_previous_day1
- relative_humidity_2m vs relative_humidity_2m_previous_day1
- wind_speed_10m vs wind_speed_10m_previous_day1
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed/forecast_correction_pairs")
DATA_DIR.mkdir(parents=True, exist_ok=True)

CITIES = {
    "delhi": {"lat": 28.6139, "lon": 77.2090, "elevation": 216.0, "zone": "composite_north"},
    "mumbai": {"lat": 19.0760, "lon": 72.8777, "elevation": 14.0, "zone": "warm_humid_west"},
    "kolkata": {"lat": 22.5726, "lon": 88.3639, "elevation": 9.0, "zone": "warm_humid_east"},
    "nagpur": {"lat": 21.1458, "lon": 79.0882, "elevation": 310.0, "zone": "hot_dry_central"},
}


def get_season(month: int) -> str:
    """Classify Indian meteorological season by calendar month."""
    if month in (3, 4, 5):
        return "summer"
    elif month in (6, 7, 8, 9):
        return "monsoon"
    elif month in (10, 11):
        return "post_monsoon"
    else:
        return "winter"


def fetch_city_pairs(
    city_name: str,
    lat: float,
    lon: float,
    start_date: str = "2024-01-20",
    end_date: str = "2024-11-15",
) -> pd.DataFrame:
    """Fetch paired hourly forecast and verification from Open-Meteo Previous Runs API."""
    logger.info("Fetching paired data for %s (%s to %s)...", city_name, start_date, end_date)
    url = (
        f"https://previous-runs-api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&"
        f"hourly=temperature_2m,temperature_2m_previous_day1,"
        f"relative_humidity_2m,relative_humidity_2m_previous_day1,"
        f"precipitation,precipitation_previous_day1,"
        f"wind_speed_10m,wind_speed_10m_previous_day1"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "ClimaCore/1.0"})
    with urllib.request.urlopen(req, timeout=40) as resp:
        data = json.loads(resp.read().decode())

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "t2m_obs": hourly["temperature_2m"],
        "t2m_fcst": hourly["temperature_2m_previous_day1"],
        "rh_obs": hourly["relative_humidity_2m"],
        "rh_fcst": hourly["relative_humidity_2m_previous_day1"],
        "precip_obs": hourly["precipitation"],
        "precip_fcst": hourly["precipitation_previous_day1"],
        "ws_obs": hourly["wind_speed_10m"],
        "ws_fcst": hourly["wind_speed_10m_previous_day1"],
    })
    df["city"] = city_name
    df["latitude"] = lat
    df["longitude"] = lon
    df["elevation"] = CITIES[city_name]["elevation"]
    df["zone"] = CITIES[city_name]["zone"]
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["month"] = df["timestamp"].dt.month
    df["season"] = df["month"].apply(get_season)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_year"] = df["timestamp"].dt.day_of_year
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["lead_days"] = 1

    # Drop any remaining unarchived hours
    df = df.dropna().reset_index(drop=True)
    return df


def aggregate_daily(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly paired data to daily metrics for operational forecasting."""
    grouped = df_hourly.groupby(["city", "date", "latitude", "longitude", "elevation", "zone", "season"])
    daily = grouped.agg(
        t2m_max_obs=("t2m_obs", "max"),
        t2m_max_fcst=("t2m_fcst", "max"),
        t2m_min_obs=("t2m_obs", "min"),
        t2m_min_fcst=("t2m_fcst", "min"),
        t2m_mean_obs=("t2m_obs", "mean"),
        t2m_mean_fcst=("t2m_fcst", "mean"),
        precip_obs=("precip_obs", "sum"),
        precip_fcst=("precip_fcst", "sum"),
        rh_mean_obs=("rh_obs", "mean"),
        rh_mean_fcst=("rh_fcst", "mean"),
        ws_mean_obs=("ws_obs", "mean"),
        ws_mean_fcst=("ws_fcst", "mean"),
        day_of_year=("day_of_year", "first"),
        sin_doy=("sin_doy", "first"),
        cos_doy=("cos_doy", "first"),
        lead_days=("lead_days", "first"),
    ).reset_index()

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["city", "date"]).reset_index(drop=True)
    return daily


def main() -> None:
    all_hourly = []
    for city_name, meta in CITIES.items():
        try:
            df_city = fetch_city_pairs(city_name, meta["lat"], meta["lon"])
            all_hourly.append(df_city)
        except Exception as e:
            logger.error("Failed to fetch %s: %s", city_name, e)

    if not all_hourly:
        raise RuntimeError("No data could be fetched from Previous Runs API.")

    combined_hourly = pd.concat(all_hourly, ignore_index=True)
    combined_daily = aggregate_daily(combined_hourly)

    hourly_path = DATA_DIR / "hourly_forecast_obs_pairs.parquet"
    daily_path = DATA_DIR / "daily_forecast_obs_pairs.parquet"

    combined_hourly.to_parquet(hourly_path, index=False)
    combined_daily.to_parquet(daily_path, index=False)

    logger.info("Saved %d hourly pairs (0 NaNs) to %s", len(combined_hourly), hourly_path)
    logger.info("Saved %d daily pairs (0 NaNs) to %s", len(combined_daily), daily_path)


if __name__ == "__main__":
    main()
