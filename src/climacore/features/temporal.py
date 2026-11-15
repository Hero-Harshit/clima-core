"""
Temporal feature engineering for Clima-Core.
Computes calendar features and cyclical sine/cosine transformations.
"""

import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Add time features from date column:
    - day of year, month, day of week
    - cyclical sine and cosine for day of year (period 365.25)
    - cyclical sine and cosine for month (period 12)
    """
    df = df.copy()
    dt = pd.to_datetime(df[date_col])

    df["day_of_year"] = dt.dt.day_of_year
    df["month"] = dt.dt.month
    df["day_of_week"] = dt.dt.day_of_week
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical day of year
    df["sin_day_of_year"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25).astype(np.float32)
    df["cos_day_of_year"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25).astype(np.float32)

    # Cyclical month
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0).astype(np.float32)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0).astype(np.float32)

    return df
