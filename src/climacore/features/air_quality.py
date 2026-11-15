"""
Feature engineering pipeline specifically for the AirQuality module.
Builds leakage-safe tabular training tables combining air quality history,
meteorological context, spatial metadata, and calendar dynamics.
"""

from typing import List, Tuple
import pandas as pd

from climacore.features.temporal import add_temporal_features
from climacore.features.lags import add_lag_features
from climacore.features.rolling import add_rolling_features


def build_air_quality_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Construct all time-safe features for next-day PM2.5 prediction.
    Returns the enriched DataFrame and the list of feature column names.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["city", "date"]).reset_index(drop=True)

    # 1. Temporal features
    df = add_temporal_features(df, date_col="date")

    # 2. Lag features for pollutants and weather
    pollutant_lags = ["pm2_5_mean", "pm10_mean", "no2_mean", "aod_mean"]
    df = add_lag_features(
        df,
        group_col="city",
        date_col="date",
        target_cols=pollutant_lags,
        lags=[1, 2, 3, 7],
    )

    weather_lags = ["wind_speed_10m_max", "precipitation_sum", "temperature_2m_mean", "relative_humidity_2m_mean"]
    df = add_lag_features(
        df,
        group_col="city",
        date_col="date",
        target_cols=weather_lags,
        lags=[1, 2],
    )

    # 3. Rolling window features
    df = add_rolling_features(
        df,
        group_col="city",
        date_col="date",
        target_cols=["pm2_5_mean", "pm10_mean"],
        windows=[3, 7, 14],
        shift_before_rolling=0,
    )

    df = add_rolling_features(
        df,
        group_col="city",
        date_col="date",
        target_cols=["wind_speed_10m_max", "precipitation_sum"],
        windows=[3, 7],
        shift_before_rolling=0,
    )

    # 4. Target variable: Next day PM2.5 mean (t+1)
    # Shifted by -1 within each city so row at t contains features at t and target at t+1
    df["target_pm2_5_next_day"] = df.groupby("city")["pm2_5_mean"].shift(-1)

    # Convert categorical region/city to category dtypes for LightGBM
    df["city_cat"] = df["city"].astype("category")
    if "region" in df.columns:
        df["region_cat"] = df["region"].astype("category")

    # List of selected feature columns (excluding identifiers, dates, targets)
    excluded_cols = {
        "date",
        "city",
        "state",
        "region",
        "target_pm2_5_next_day",
        "india_aqi_category",
        "india_aqi_pm_category",
        "dominant_pollutant",
    }

    feature_cols = [c for c in df.columns if c not in excluded_cols]

    return df, feature_cols
