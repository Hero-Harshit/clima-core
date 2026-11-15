"""
Feature engineering pipeline specifically for the RainRisk module.
Builds time-safe features combining precipitation lags, accumulation windows,
atmospheric moisture, wind dynamics, and monsoon/calendar cycles.
"""

from typing import List, Tuple
import pandas as pd
import numpy as np

from climacore.features.temporal import add_temporal_features
from climacore.features.lags import add_lag_features
from climacore.features.rolling import add_rolling_features


def build_rain_risk_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Construct all time-safe features for next-day precipitation prediction.
    Returns the enriched DataFrame and the list of feature column names.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["city", "date"]).reset_index(drop=True)

    # 1. Temporal & Monsoon Features
    df = add_temporal_features(df, date_col="date")

    # Indian Monsoon Seasonality Indicator
    # Monsoon (JJAS): June (6) to September (9)
    # Post-Monsoon (ON): October (10) to November (11)
    # Winter (DJF): December (12) to February (2)
    # Pre-Monsoon / Summer (MAM): March (3) to May (5)
    df["is_monsoon_season"] = df["month"].isin([6, 7, 8, 9]).astype(int)
    df["is_post_monsoon"] = df["month"].isin([10, 11]).astype(int)

    # 2. Lag Features
    precip_lags = [
        "precipitation_sum",
        "precipitation_hours",
        "relative_humidity_2m_mean",
        "temperature_2m_mean",
        "temperature_2m_min",
        "temperature_2m_max",
        "wind_speed_10m_max",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
    ]
    df = add_lag_features(
        df,
        group_col="city",
        date_col="date",
        target_cols=precip_lags,
        lags=[1, 2, 3, 7],
    )

    # 3. Rolling Accumulation & Maximums
    # Rolling sums of precipitation (3d, 7d, 14d) to represent soil moisture & recent wetness
    df = df.sort_values(by=["city", "date"]).copy()
    grouped_precip = df.groupby("city")["precipitation_sum"]

    for w in [3, 7, 14]:
        df[f"precipitation_sum_roll_sum_{w}d"] = (
            grouped_precip.rolling(window=w, min_periods=1).sum().reset_index(level=0, drop=True)
        )
        df[f"precipitation_sum_roll_max_{w}d"] = (
            grouped_precip.rolling(window=w, min_periods=1).max().reset_index(level=0, drop=True)
        )
        df[f"precipitation_hours_roll_sum_{w}d"] = (
            df.groupby("city")["precipitation_hours"].rolling(window=w, min_periods=1).sum().reset_index(level=0, drop=True)
        )

    # 4. Target Variables (Next-Day, strictly shifted by -1 within each city)
    # Precipitation amount in mm
    df["target_precipitation_next_day"] = df.groupby("city")["precipitation_sum"].shift(-1)
    # Binary occurrence: Rain >= 1.0 mm
    df["target_rain_occurrence_next_day"] = (df["target_precipitation_next_day"] >= 1.0).astype(int)
    # Official IMD Rainy Day: Rain >= 2.5 mm
    df["target_imd_rainy_day_next_day"] = (df["target_precipitation_next_day"] >= 2.5).astype(int)
    # Heavy Rain indicator: Rain >= 15.6 mm
    df["target_heavy_rain_next_day"] = (df["target_precipitation_next_day"] >= 15.6).astype(int)

    # Categoricals for LightGBM
    df["city_cat"] = df["city"].astype("category")
    if "region" in df.columns:
        df["region_cat"] = df["region"].astype("category")

    # Excluded columns from input feature set
    excluded_cols = {
        "date",
        "city",
        "state",
        "region",
        "target_precipitation_next_day",
        "target_rain_occurrence_next_day",
        "target_imd_rainy_day_next_day",
        "target_heavy_rain_next_day",
    }

    feature_cols = [c for c in df.columns if c not in excluded_cols]

    return df, feature_cols
