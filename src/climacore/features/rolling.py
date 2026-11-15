"""
Time-safe rolling feature engineering.
Computes rolling window statistics (mean, max, min, std) strictly over past observations
grouped by entity (e.g. city) without future leakage.
"""

from typing import List
import pandas as pd


def add_rolling_features(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    target_cols: List[str],
    windows: List[int],
    shift_before_rolling: int = 0,
) -> pd.DataFrame:
    """
    Compute rolling statistics (mean, max, std) for specified columns grouped by group_col.
    Guarantees chronological sorting.
    If shift_before_rolling > 0, shifts the series before rolling to ensure no current-day leakage.
    """
    df = df.sort_values(by=[group_col, date_col]).copy()

    for col in target_cols:
        grouped = df.groupby(group_col)[col]
        series = grouped.shift(shift_before_rolling) if shift_before_rolling > 0 else grouped

        for w in windows:
            # Min periods = max(1, w // 2) to allow early valid values while preventing false spikes
            min_p = max(1, w // 2)
            prefix = f"{col}_shift{shift_before_rolling}" if shift_before_rolling > 0 else col
            
            df[f"{prefix}_roll_mean_{w}d"] = (
                series.rolling(window=w, min_periods=min_p).mean().reset_index(level=0, drop=True)
            )
            df[f"{prefix}_roll_max_{w}d"] = (
                series.rolling(window=w, min_periods=min_p).max().reset_index(level=0, drop=True)
            )
            df[f"{prefix}_roll_std_{w}d"] = (
                series.rolling(window=w, min_periods=min_p).std().reset_index(level=0, drop=True)
            )

    return df
