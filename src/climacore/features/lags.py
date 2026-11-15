"""
Time-safe lag feature engineering.
Ensures grouping by entity (e.g. city) and strictly chronological ordering
without any future information leakage.
"""

from typing import List
import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    target_cols: List[str],
    lags: List[int],
) -> pd.DataFrame:
    """
    Compute lag features for specified columns grouped by group_col.
    Data is guaranteed to be chronologically sorted before lagging.
    """
    df = df.sort_values(by=[group_col, date_col]).copy()

    for col in target_cols:
        for lag in lags:
            feature_name = f"{col}_lag_{lag}"
            df[feature_name] = df.groupby(group_col)[col].shift(lag)

    return df
